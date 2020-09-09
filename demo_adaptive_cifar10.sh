for attack_method in FGSM
do
for epsilon in  0.00784 0.03137 0.06275
do 
for iters in  3
do 
	python main_cifar10_adaptive.py  --epsilon $epsilon --test_samples 500  --attack_method $attack_method --nb_iters $iters | tee ./logs/adaptive/cifar10_$attack_method\_$epsilon.txt
done
done
done
