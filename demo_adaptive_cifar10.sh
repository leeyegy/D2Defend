for attack_method in NONE
do
for epsilon in  0.0
do 
for iters in  3
do 
	python main_cifar10_adaptive.py  --epsilon $epsilon --test_samples 500  --attack_method $attack_method --nb_iters $iters | tee ./logs/adaptive/v16/cifar10_new_$attack_method\_$epsilon.txt
done
done
done
