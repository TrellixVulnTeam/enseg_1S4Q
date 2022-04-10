# ensegv5：新学期写的一个框架，结合图像翻译和语义分割。特点是图像翻译时从大图像翻译成小图像
* enseg_v5_swin_upergen_h256w512_bs4_80k_nc.py：生成器使用mmcv提供的upernet架构
* enseg_v5_swin_unetgen_h256w512_bs4_80k_nc.py：对upernet架构改进：原有的upernet是fpn逐级处理后相邻级相加，现在改成了先相加后处理，更类似于unet的架构
* 



'''###____pretty_text____###'''



'''
False to Parse'''
