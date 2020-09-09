for attack_method in PGD 
do
for epsilon in  0.00784 0.03137 0.06275
do 
for iters in  3 
do 
	python main_imagenet_adaptive.py --epsilon $epsilon --test_samples 1000  --attack_method $attack_method --nb_iters $iters | tee ./logs/adaptive/tiny_imagenet/v16_1.0/$attack_method\_$epsilon\_1000_iters_$iters.txt
done
done
done
for attack_method in NONE
do
for epsilon in  0.0
do 
for iters in  3 
do 
	python main_imagenet_adaptive.py --epsilon $epsilon --test_samples 1000  --attack_method $attack_method --nb_iters $iters | tee ./logs/adaptive/tiny_imagenet/v16_1.0/$attack_method\_$epsilon\_1000_iters_$iters.txt
done
done
done
