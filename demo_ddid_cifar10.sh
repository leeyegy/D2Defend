for attack_method in   DeepFool
do
for epsilon in 0.06275
do 
	python main_cifar10_pgd.py --epsilon $epsilon --test_samples 10000  --attack_method $attack_method | tee ./logs/adv_training_defence/pgd_8_$attack_method\_$epsilon.txt
done
done
