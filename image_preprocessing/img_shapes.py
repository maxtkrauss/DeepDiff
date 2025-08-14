from tifffile import imread

cubert = r"D:\fakenotes_8-14\cubert\image_1_cubert.tif"
thorlabs = r"D:\fakenotes_8-14\thorlabs\image_1_thorlabs.tif"

cubert_img = imread(cubert)
thorlabs_img = imread(thorlabs)

print("Cubert image shape:", cubert_img.shape)
print("Thorlabs image shape:", thorlabs_img.shape)