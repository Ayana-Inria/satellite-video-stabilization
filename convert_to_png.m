clear
directory_n = 'C:\Users\caguilar\Desktop\Data\US_Airforce\';
folder_n = 'WPAFB-21Oct2009-TRAIN_NITF_001\';

if(~exist(folder_n))
   mkdir(folder_n);
   mkdir([folder_n(1:end - 1) '_cropped']);
end

files = {'20091021203201-01000605-VIS.ntf.r1'; '20091021203202-01000606-VIS.ntf.r1'; '20091021203202-01000607-VIS.ntf.r1'; '20091021203203-01000608-VIS.ntf.r1'; '20091021203204-01000609-VIS.ntf.r1'};
for number=1:5
    disp(number);
    file_n = files{number}; %['20091021203201-0100060' num2str(number + 5) '-VIS.ntf.r5'];
    disp(file_n);
    path_n = [directory_n folder_n file_n];
    X = nitfread(path_n);
    imwrite(X, [folder_n '/image' num2str(number) '.png']);
    [r, c] = size(X);
    xo = int16(0.68 * r);
    yo = int16(0.319 * c);
    wz = int16(0.05 * r);
    Xp = X(xo:xo + wz, yo:yo + wz);
    imwrite(Xp, [folder_n(1:end - 1)  '_cropped/'  'image' num2str(number) '.png']);
end

