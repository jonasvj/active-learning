#!/bin/sh
### General options
### –- specify queue --
#BSUB -q "gpuv100 gpua100 gpua10 gpua40"
### -- set the job Name --
#BSUB -J test_job
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 00:10
# request 5GB of system-memory
#BSUB -R "rusage[mem=1GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
###BSUB -B
### -- send notification at completion--
###BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o ~/active-learning/gpu_%J.out
#BSUB -e ~/active-learning/gpu_%J.err
# -- end of LSF options --

cd ~/
source .virtualenvs/al/bin/activate
cd ~/active-learning

echo "Submit to several queues test"
sleep 1000000000000000000