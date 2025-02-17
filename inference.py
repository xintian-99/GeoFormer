from argparse import Namespace
import torch
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from model.loftr_src.loftr.utils.cvpr_ds_config import default_cfg
from model.full_model import GeoFormer as GeoFormer_

from eval_tool.immatch.utils.data_io import load_gray_scale_tensor_cv
from model.geo_config import default_cfg as geoformer_cfg

class GeoFormer():
    def __init__(self, imsize, match_threshold, no_match_upscale=False, ckpt=None, device='cuda'):

        self.device = device
        self.imsize = imsize
        self.match_threshold = match_threshold
        self.no_match_upscale = no_match_upscale

        # Load model
        conf = dict(default_cfg)
        conf['match_coarse']['thr'] = self.match_threshold
        geoformer_cfg['coarse_thr'] = self.match_threshold
        self.model = GeoFormer_(conf)
        ckpt_dict = torch.load(ckpt, map_location=torch.device('cpu'))
        if 'state_dict' in ckpt_dict:
            ckpt_dict = ckpt_dict['state_dict']
        self.model.load_state_dict(ckpt_dict, strict=False)
        self.model = self.model.eval().to(self.device)

        # Name the method
        self.ckpt_name = ckpt.split('/')[-1].split('.')[0]
        self.name = f'GeoFormer_{self.ckpt_name}'
        if self.no_match_upscale:
            self.name += '_noms'
        print(f'Initialize {self.name}')

    def change_deivce(self, device):
        self.device = device
        self.model.to(device)
    def load_im(self, im_path, enhanced=False):
        return load_gray_scale_tensor_cv(
            im_path, self.device, imsize=self.imsize, dfactor=8, enhanced=enhanced, value_to_scale=min
        )

    def match_inputs_(self, gray1, gray2, is_draw=False):

        batch = {'image0': gray1, 'image1': gray2}
        with torch.no_grad():
            batch = self.model(batch)
        kpts1 = batch['mkpts0_f'].cpu().numpy()
        kpts2 = batch['mkpts1_f'].cpu().numpy()
        def draw():
            import matplotlib.pyplot as plt
            import cv2
            import numpy as np
            plt.figure(dpi=200)
            kp0 = kpts1
            kp1 = kpts2
            # if len(kp0) > 0:
            kp0 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp0]
            kp1 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp1]
            matches = [cv2.DMatch(_trainIdx=i, _queryIdx=i, _distance=1, _imgIdx=-1) for i in
                       range(len(kp0))]

            show = cv2.drawMatches((gray1.cpu()[0][0].numpy() * 255).astype(np.uint8), kp0,
                                   (gray2.cpu()[0][0].numpy() * 255).astype(np.uint8), kp1, matches,
                                   None)
            plt.imshow(show)
            plt.show()
        if is_draw:
            draw()
        scores = batch['mconf'].cpu().numpy()
        matches = np.concatenate([kpts1, kpts2], axis=1)
        return matches, kpts1, kpts2, scores

    def match_pairs(self, im1_path, im2_path, cpu=False, is_draw=False):
        torch.cuda.empty_cache()
        tmp_device = self.device
        if cpu:
            self.change_deivce('cpu')
        gray1, sc1 = self.load_im(im1_path)
        gray2, sc2 = self.load_im(im2_path)

        upscale = np.array([sc1 + sc2])
        matches, kpts1, kpts2, scores = self.match_inputs_(gray1, gray2, is_draw)

        if self.no_match_upscale:
            return matches, kpts1, kpts2, scores, upscale.squeeze(0)

        # Upscale matches &  kpts
        matches = upscale * matches
        kpts1 = sc1 * kpts1
        kpts2 = sc2 * kpts2

        if cpu:
            self.change_deivce(tmp_device)
                    # Print keypoints

         

        # Save keypoints
        np.save("keypoints_image1.npy", kpts1)
        np.save("keypoints_image2.npy", kpts2)

        return matches, kpts1, kpts2, scores

def draw_keypoints(image_path, keypoints, title="Keypoints", num_points=5):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for i, k in enumerate(keypoints[:num_points]):
        x, y = int(k[0]), int(k[1])  # 假设是 (x, y)
        cv2.circle(image_color, (x, y), 5, (0, 255, 0), -1)  # 绿色点
        cv2.putText(image_color, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

    plt.figure(figsize=(6, 6))
    plt.imshow(image_color[:, :, ::-1])  # OpenCV 是 BGR，Matplotlib 需要 RGB
    plt.title(title)
    plt.axis("off")
    plt.show()
    

    
def process_fire_images(image_dir, output_dir, geoformer):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    images = sorted(os.listdir(image_dir))
    pairs = {}
    
    for img in images:
        if img.endswith('.jpg'):
            base_name = img[:3]  # Extract first three characters
            suffix = img[3:]
            if suffix == '_1.jpg':
                pairs.setdefault(base_name, [None, None])[0] = img
            elif suffix == '_2.jpg':
                pairs.setdefault(base_name, [None, None])[1] = img
    
    for base_name, (im1, im2) in pairs.items():
        if im1 and im2:
            im1_path = os.path.join(image_dir, im1)
            im2_path = os.path.join(image_dir, im2)
            # print(f'Processing pair: {im1}, {im2}')
            matches, kpts1, kpts2, scores = geoformer.match_pairs(im1_path, im2_path)
            #print the ketpoint number
            # print(f'Keypoints for {base_name}: {len(kpts1)}')
            print(f'Keypoints for {base_name}: {len(kpts1)}')
            # Save keypoints in different formats
            base_output_path = os.path.join(output_dir, base_name)
            # np.save(f'{base_output_path}_kpts1.npy', kpts1)
            # np.save(f'{base_output_path}_kpts2.npy', kpts2)
            # np.save(f'{base_output_path}_matches.npy', matches)
            # np.savetxt(f'{base_output_path}_1.txt', kpts1, fmt='%.6f')
            # np.savetxt(f'{base_output_path}_2.txt', kpts2, fmt='%.6f')
            np.savetxt(f'{base_output_path}_1_2.txt', matches, fmt='%.6f')
            #save to csv and only keep 6 decimals
            # np.savetxt(f'{base_output_path}_1.csv', kpts1, delimiter=",", fmt='%.6f')
            # np.savetxt(f'{base_output_path}_2.csv', kpts2, delimiter=",", fmt='%.6f')
            np.savetxt(f'{base_output_path}_1_2.csv', matches, delimiter=",", fmt='%.6f')

            # print(f'Saved keypoints for {base_name}')
            # draw_keypoints(im1_path, kpts1, title=f"{base_name} - Image 1 Keypoints")
            # draw_keypoints(im2_path, kpts2, title=f"{base_name} - Image 2 Keypoints")


if __name__ == "__main__":
    image_dir = "E:/Github/GeoFormer/data/datasets/FIRE/Images/"
    output_dir = "E:/Github/GeoFormer/keypoints/fire/"
    
    geoformer = GeoFormer(640, 0.3, no_match_upscale=False, ckpt='saved_ckpt/geoformer.ckpt', device='cuda')
    process_fire_images(image_dir, output_dir, geoformer)


# g = GeoFormer(640, 0.2, no_match_upscale=False, ckpt='saved_ckpt/geoformer.ckpt', device='cuda')
# g.match_pairs('E:/Github/GeoFormer/data/datasets/FIRE/Images/A01_1.jpg', 'E:/Github/GeoFormer/data/datasets/FIRE/Images/A01_2.jpg', is_draw=True)
