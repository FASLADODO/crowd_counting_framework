import piq
import torch 

a = torch.rand(256,256)
a_copy = a.detach().clone()
b = torch.rand(256,256)

aa = piq.ssim(a, a_copy)
ab = piq.ssim(a, b)
print(aa)
print(ab)

###########
paa = piq.psnr(a, a, data_range=100)
pab = piq.psnr(a, b, data_range=b.max())

print(paa)
print(pab)

#################
# batch
a_batch = torch.rand(10, 1, 256, 256)
b_batch = torch.rand(10, 1, 256, 256)

aa_batchn = piq.ssim(a_batch, b_batch)
print(aa_batchn)
ss_sum = 0
for i in range(10):
    a_item = a_batch[i]
    b_item = b_batch[i]
    ssim_item = piq.ssim(a_item, b_item)
    ss_sum += ssim_item
print(ss_sum/10)

paa_batchn = piq.psnr(a_batch, b_batch)
print(paa_batchn)

pss_sum = 0
for i in range(10):
    a_item = a_batch[i]
    b_item = b_batch[i]
    p_item = piq.psnr(a_item, b_item)
    pss_sum += p_item
print(pss_sum/10)