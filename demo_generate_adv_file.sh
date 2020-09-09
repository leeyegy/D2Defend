for epsilon in  0.06275
do
for attack_method in DeepFool
do 
python data_generator.py --attack_method $attack_method   --epsilon $epsilon
done
done 
