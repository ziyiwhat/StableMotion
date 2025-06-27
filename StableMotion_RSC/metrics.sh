python metric.py --results results --gts ./Real_RS/test/jpg_gt >> metrics.txt
python -m pytorch_fid ./results ./Real_RS/test/jpg_gt >> Ours_40000_metrics.txt
pyiqa lpips -t ./results -r ./Real_RS/test/jpg_gt >> Ours_40000_metrics.txt
pyiqa topiq_fr -t ./results -r ./Real_RS/test/jpg_gt >> Ours_40000_metrics.txt
pyiqa fid -t ./results -r ./Real_RS/test/jpg_gt >> Ours_40000_metrics.txt
