for attack in FGSM PGD Momentum CW DeepFool
do 
	for epsilon in 0.03137 0.06275 0.00784
	do 
	python data_generator.py --attack_method $attack --epsilon $epsilon 
done 
done 
