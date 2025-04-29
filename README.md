## Run Trainers with Different Gradient Computation Methods
1. The trainer files are divided into three categories and can be found in `./trainers` directory:
	
    a. Backpropagation-based Trainers
	    
        `backprop.py` (BP-Vanilla, BP-Checkpointing, BP-Accumulate) can be executed through `bash run_backprop_job.sh`.
    
    b. Zero-order-based Trainers
        
        `zero_finite_differences.py` (ZO-Vanilla, ZO-Accumulate, ZO-Multiple, ZO-Adaptive), `svrg_zero_finite_differences.py` (ZO-SVRG), and `sparse_zero_finite_differences.py` (ZO-Sparse) can be all executed through `bash run_zo_job.sh`.
    
    c. Forward-mode AD-based Trainers
        
        `forward_mode_ad_beta.py` (FmAD-Vanilla, FmAD-Multiple, FmAD-Adaptive), `more_forward_mode_ad_beta.py` (FmAD-Accumulate), `svrg_forward_mode_ad_beta` (FmAD-SVRG) and `sparse_forward_mode_ad_beta.py` (FmAD-Sparse) can be all executed through `bash run_zo_job.sh`.
 
 2.    All the hyperparameters will be set through the `.sh` files for their corresponding gradient computation method. 
 
 3. The results will be saved in an automatically created `./results` directory.
