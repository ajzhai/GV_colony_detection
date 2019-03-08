% Script for displaying the data from a singlescan of two plates. 
% This code is adapted from the main script that stiches together all the
% individual image planes, as that has already been done for this data.
% These data were acquired with the L10-4v probe using 19.5° xAM at 40V
% with a 1-cycle transmit waveform at 10 MHz.

% Comment out the lines below after variables are loaded to reduce run time
load('GVpBScanVolumeData.mat'); % load the following variables:
if iscell(imVolAM)
    imVolAM = imVolAM{1};
end
if iscell(imVolBmode)
    imVolBmode = imVolBmode{1};
end

% imVolBmode -- The linear-scale 3D xB-mode image volume, indexed (y,x,z)
% imVolAM -- The linear-scale 3D xAM image volume, indexed (y,x,z)
% y -- Depth / axial position (mm), going into the plate
% z -- Latteral position (mm), going across the plate.
% x -- Transverse position (mm), going down the plate. Each position in this 
% direction is a separate image plane
% Nzlines -- The number of image stacks concatenated in the lateral (z)
% dimension. Each depth image plane is formed from 64 ray lines, so the
% length of z is 64*Nzlines.

%% Choose the integration region
% Display depth plane for choosing depth cross section over which to take 
% the max for forming the 2D image. Ignore the lateral dimension of the rectangle
close all

figure 
imagesc(20*log10(squeeze(imVolBmode(:,40,:))))
title('Click & drag to choose displayed depth')
r = imrect;
r = round(getPosition(r));
integDepth = r(2):r(2)+r(4);

%% Process volume images to get 2D plate image

% 2D linear scale images
imAM = squeeze(max(imVolAM(integDepth,:,:),[],1));
imBmode = squeeze(max(imVolBmode(integDepth,:,:),[],1));

% Convert to dB (log scale) for display purposes
imBmode_dB = 20*log10(imBmode/max(imBmode(:)));
imAM_dB = 20*log10(imAM/max(imAM(:)));

% Define a normalization function based on robust statistics
normalize = @(x) (x - median(x(:))) / iqr(x(:))...
    - min((x(:) - median(x(:))) / iqr(x(:)));
% Get the "ratio" of the xAM to xB-mode to control for colonies having high
% residual xAM signal due to strong signal in xB-mode
imRatio = (1+normalize(imAM)) ./ (1+normalize(imBmode));
imRatio_dB = 20*log10(imRatio);

%% Plot the xB-mode, xAM, and ratio images for each scan
figure('Position',[1 1 1440 804])
colormap hot

subplot(3,1,1)
imagesc(z,x,imBmode_dB,[-50 0])
axis image
colorbar
title('55V')
ylabel('xB-mode','Fontsize',14);

subplot(3,1,2)
imagesc(z,x,imAM_dB,[-50 0])
axis image
colorbar
ylabel('xAM','Fontsize',14);

subplot(3,1,3)
imagesc(z,x,imRatio_dB,[0 inf])
axis image
colorbar
ylabel('xAM:xB-mode ratio','Fontsize',14);

minVal = min(min(imAM_dB));
maxVal = max(max(imAM_dB));
imAM_norm = (imAM_dB - minVal) ./ (maxVal - minVal);
imwrite(imAM_norm, 'imAM_.png')
minVal = min(min(imBmode_dB));
maxVal = max(max(imBmode_dB));
imBmode_norm = (imBmode_dB - minVal) ./ (maxVal - minVal);
imwrite(imBmode_norm, 'imBmode_.png')
minVal = min(min(imRatio_dB));
maxVal = max(max(imRatio_dB));
imRatio_norm = (imRatio_dB - minVal) ./ (maxVal - minVal);
imwrite(imRatio_norm, 'imRatio_.png')

% Use the 2D linear data for export/processing
save('Linear2DPlateScanData_.mat','imBmode','imAM','imRatio','z','x')