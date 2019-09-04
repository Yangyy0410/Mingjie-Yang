#$ -cwd
#$ -V
#$ -l h_rt=48:00:00
#$ -l h_vmem=128G
#$ -l coproc_p100=2
python3 UniLSTM3L64.py
