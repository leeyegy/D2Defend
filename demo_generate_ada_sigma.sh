for attack in FGSM
do 
	for epsilon in 0.00784 0.03137 0.06275
	do 
	python data_generator.py --task g_adaptive_sigma --attack_method $attack --epsilon $epsilon 
done 
done 
