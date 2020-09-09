for attack in NONE
do 
	for epsilon in 0.0
	do 
	python data_generator.py --attack_method $attack --epsilon $epsilon 
done 
done 
