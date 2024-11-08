# DFNO
python train.py datamodule=vorticity_Re_16000 datamodule.model=DFNO datamodule.b_train_val_test=[64,10,1] model=DFNO
python train.py datamodule=vorticity_Re_32000 datamodule.model=DFNO datamodule.b_train_val_test=[64,10,1] model=DFNO
python train.py datamodule=kinetic_energy datamodule.model=DFNO datamodule.b_train_val_test=[64,10,1] model=DFNO
python train.py datamodule=temperature datamodule.model=DFNO datamodule.b_train_val_test=[64,10,1] model=DFNO
python train.py datamodule=water_vapor datamodule.model=DFNO datamodule.b_train_val_test=[64,10,1] model=DFNO

# EDSR
python train.py datamodule=vorticity_Re_16000 datamodule.model=EDSR datamodule.b_train_val_test=[64,10,1] model=EDSR
python train.py datamodule=vorticity_Re_32000 datamodule.model=EDSR datamodule.b_train_val_test=[64,10,1] model=EDSR
python train.py datamodule=kinetic_energy datamodule.model=EDSR datamodule.b_train_val_test=[64,10,1] model=EDSR
python train.py datamodule=temperature datamodule.model=EDSR datamodule.b_train_val_test=[64,10,1] model=EDSR
python train.py datamodule=water_vapor datamodule.model=EDSR datamodule.b_train_val_test=[64,10,1] model=EDSR

# ESPCN
python train.py datamodule=vorticity_Re_16000 datamodule.model=ESPCN datamodule.b_train_val_test=[64,10,1] model=ESPCN
python train.py datamodule=vorticity_Re_32000 datamodule.model=ESPCN datamodule.b_train_val_test=[64,10,1] model=ESPCN
python train.py datamodule=kinetic_energy datamodule.model=ESPCN datamodule.b_train_val_test=[64,10,1] model=ESPCN
python train.py datamodule=temperature datamodule.model=ESPCN datamodule.b_train_val_test=[64,10,1] model=ESPCN
python train.py datamodule=water_vapor datamodule.model=ESPCN datamodule.b_train_val_test=[64,10,1] model=ESPCN

# SRCNN
python train.py datamodule=vorticity_Re_16000 datamodule.model=SRCNN datamodule.b_train_val_test=[64,10,1] model=SRCNN
python train.py datamodule=vorticity_Re_32000 datamodule.model=SRCNN datamodule.b_train_val_test=[64,10,1] model=SRCNN
python train.py datamodule=kinetic_energy datamodule.model=SRCNN datamodule.b_train_val_test=[64,10,1] model=SRCNN
python train.py datamodule=temperature datamodule.model=SRCNN datamodule.b_train_val_test=[64,10,1] model=SRCNN
python train.py datamodule=water_vapor datamodule.model=SRCNN datamodule.b_train_val_test=[64,10,1] model=SRCNN

# SwinIR
python train.py datamodule=vorticity_Re_16000 datamodule.model=SwinIR datamodule.b_train_val_test=[32,10,1] model=SwinIR model.params_model.resol_h=1024 model.params_model.resol_w=1024
python train.py datamodule=vorticity_Re_32000 datamodule.model=SwinIR datamodule.b_train_val_test=[32,10,1] model=SwinIR model.params_model.resol_h=1024 model.params_model.resol_w=1024
python train.py datamodule=kinetic_energy datamodule.model=SwinIR datamodule.b_train_val_test=[32,10,1] model=SwinIR model.params_model.resol_h=720 model.params_model.resol_w=1440
python train.py datamodule=temperature datamodule.model=SwinIR datamodule.b_train_val_test=[32,10,1] model=SwinIR model.params_model.resol_h=720 model.params_model.resol_w=1440
python train.py datamodule=water_vapor datamodule.model=SwinIR datamodule.b_train_val_test=[32,10,1] model=SwinIR model.params_model.resol_h=720 model.params_model.resol_w=1440

# WDSR
python train.py datamodule=vorticity_Re_16000 datamodule.model=WDSR datamodule.b_train_val_test=[32,10,1] model=WDSR
python train.py datamodule=vorticity_Re_32000 datamodule.model=WDSR datamodule.b_train_val_test=[32,10,1] model=WDSR
python train.py datamodule=kinetic_energy datamodule.model=WDSR datamodule.b_train_val_test=[32,10,1] model=WDSR
python train.py datamodule=temperature datamodule.model=WDSR datamodule.b_train_val_test=[32,10,1] model=WDSR
python train.py datamodule=water_vapor datamodule.model=WDSR datamodule.b_train_val_test=[32,10,1] model=WDSR

# LIIF
python train.py datamodule=vorticity_Re_16000 model=LIIF datamodule.model=LIIF datamodule.b_train_val_test=[16,10,1]
python train.py datamodule=vorticity_Re_32000 model=LIIF datamodule.model=LIIF datamodule.b_train_val_test=[16,10,1]
python train.py datamodule=kinetic_energy model=LIIF datamodule.model=LIIF datamodule.b_train_val_test=[32,20,1]
python train.py datamodule=temperature model=LIIF datamodule.model=LIIF datamodule.b_train_val_test=[32,20,1]
python train.py datamodule=water_vapor model=LIIF datamodule.model=LIIF datamodule.b_train_val_test=[32,20,1]

# MetaSR
python train.py datamodule=vorticity_Re_16000 model=MetaSR datamodule.model=MetaSR datamodule.b_train_val_test=[16,10,1]
python train.py datamodule=vorticity_Re_32000 model=MetaSR datamodule.model=MetaSR datamodule.b_train_val_test=[16,10,1]
python train.py datamodule=kinetic_energy model=MetaSR datamodule.model=MetaSR datamodule.b_train_val_test=[32,20,1]
python train.py datamodule=temperature model=MetaSR datamodule.model=MetaSR datamodule.b_train_val_test=[32,20,1]
python train.py datamodule=water_vapor model=MetaSR datamodule.model=MetaSR datamodule.b_train_val_test=[32,20,1]

# SRNO
python train.py datamodule=vorticity_Re_16000 model=SRNO datamodule.model=SRNO datamodule.b_train_val_test=[16,10,1]
python train.py datamodule=vorticity_Re_32000 model=SRNO datamodule.model=SRNO datamodule.b_train_val_test=[16,10,1]
python train.py datamodule=kinetic_energy model=SRNO datamodule.model=SRNO datamodule.b_train_val_test=[32,20,1]
python train.py datamodule=temperature model=SRNO datamodule.model=SRNO datamodule.b_train_val_test=[32,20,1]
python train.py datamodule=water_vapor model=SRNO datamodule.model=SRNO datamodule.b_train_val_test=[32,20,1]

# HiNOTE
python train.py datamodule=vorticity_Re_16000 model=HiNOTE datamodule.model=HiNOTE datamodule.b_train_val_test=[16,10,1]
python train.py datamodule=vorticity_Re_32000 model=HiNOTE datamodule.model=HiNOTE datamodule.b_train_val_test=[16,10,1]
python train.py datamodule=kinetic_energy model=HiNOTE datamodule.model=HiNOTE datamodule.b_train_val_test=[32,20,1]
python train.py datamodule=temperature model=HiNOTE datamodule.model=HiNOTE datamodule.b_train_val_test=[32,20,1]
python train.py datamodule=water_vapor model=HiNOTE datamodule.model=HiNOTE datamodule.b_train_val_test=[32,20,1]