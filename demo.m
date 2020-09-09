%img_name = {'0.0.png','1.0.png','2.0.png','3.0.png','4.0.png'}
attack_dir = {'FGSM_0.00784','FGSM_0.03137','FGSM_0.06275'}
%for j = 1:size(img_name,2);
%attack_dir = {'BIM_0.00784','BIM_0.03137','BIM_0.06275'}

for threshold = 20
for k = 1:size(attack_dir,2);
noise_level = []
sigma_level = []
level_num = zeros(5,1);
json_data = containers.Map;
for class = 0:9;
for img_name = 0:999;

name = strcat('data/cifar10_img/val/',attack_dir{k},'/',num2str(class),'/',num2str(img_name),'.0.png');

%img = double(imreaid(name)); original version
img = imread(name);
mskflg = 0; % if you want to produce the mask, put one for mskflg
%  [nlevel th] = NoiseLevel(noise);
%   fprintf('  R:%5.2f G:%5.2f B:%5.2f\n', nlevel(1), nlevel(2), nlevel(3) );
  
% level = [0,5,10,20,40];
level = [0];
for i=1:size(level,2);

%high-frequency 
HIGH = [];
for c=1:3
    I = img(:,:,c);
    J = dct2(I);
    
    ori_ = J;
    J(abs(J) < threshold) = 0; 
    H_dct = ori_ - J;
    HIGH(:,:,c) = idct2(H_dct);
end
img = double(HIGH);








 noise = img + randn(size(img)) * level(i);
 tic;
 [nlevel th] = NoiseLevel(noise);
 t=toc;
 noise_level = [noise_level,mean(nlevel)];




%sigma_level mapping
if mean(nlevel) <=1.90
sigma_level(img_name+1,class+1)=12;
level_num(1) = level_num(1)+ 1;
elseif mean(nlevel)<=3.05
sigma_level(img_name+1,class+1)=23;
level_num(2) = level_num(2)+ 1;
elseif mean(nlevel)<=3.25
sigma_level(img_name+1,class+1)=28;
level_num(3) = level_num(3)+ 1;
elseif mean(nlevel)<=3.70
sigma_level(img_name+1,class+1)=30;
level_num(4) = level_num(4)+ 1;
else
sigma_level(img_name+1,class+1)=35;
level_num(5) = level_num(5)+ 1;
end



 %fprintf('True: %5.2f  R:%5.2f G:%5.2f B:%5.2f mean: %5.2f \n', level(i), nlevel(1), nlevel(2), nlevel(3),mean(nlevel) );
 %fprintf('Calculation time: %5.2f [sec]\n\n', t );

 if( mskflg )
   msk = WeakTextureMask( noise, th );
   imwrite(uint8(msk*255), sprintf('msk%02d.png', level(i)));
 end
end
end

json_data(num2str(class))= sigma_level(:,class+1);

end
 fprintf('threshold:%d attack: %s mean: %5.2f max:%5.2f min:%5.2f var:%5.2f median:%5.2f \n',threshold,attack_dir{k}, mean(noise_level),max(noise_level),min(noise_level) ,var(noise_level),median(noise_level));
 fprintf('sigma=12:%d sigma=23:%d  sigma=30 :%d ,sigma=35:%d\n',level_num(1),level_num(2),level_num(3),level_num(4));
 json_text = jsonencode(json_data);
 
%save file
fid=fopen(strcat('new_',attack_dir{k},'.json'),'wt');
fprintf(fid,json_text);
fclose(fid);
filename = strcat(attack_dir{k},'.mat')
save(filename,'noise_level');
end
end % end for first 

