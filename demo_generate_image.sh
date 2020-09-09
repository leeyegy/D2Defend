for attack in FGSM 
do 
	for epsilon in 0.03137 0.06275 0.00784
	do 
	python data_generator.py --task g_img --attack_method $attack --epsilon $epsilon 
done 
done 
