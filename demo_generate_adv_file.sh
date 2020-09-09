for epsilon in  0.06275 0.00784 0.03137
do
for attack_method in FGSM
do 
    python data_generator.py --task g_adv --attack_method $attack_method   --epsilon $epsilon
done
done 
