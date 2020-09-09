for attack_method in FGSM 
do
for epsilon in  0.03137
do 
for sigma in 33 34 35 36 32 31 37
do
	python main_imagenet.py --epsilon $epsilon --test_samples 500 --sigma $sigma --attack_method $attack_method | tee ./logs/tiny_Imagenet_resnet50_sigma_$sigma\__$attack_method\_$epsilon\_500.txt
done
done
done
