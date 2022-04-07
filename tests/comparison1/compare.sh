for i in 1 5 10 
do

python compare.py --num_subjects=$i --waveform='mixing' 
python compare.py --num_subjects=$i --waveform='square' 
python compare.py --num_subjects=$i --waveform='sine'  

done
python visualize_compare.py --waveform='mixing'
python visualize_compare.py --waveform='square'
python visualize_compare.py --waveform='sine'