%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ENG EC 414 (Ishwar) Spring 2022
% HW 4
% <Your full name and BU email>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc;
rng('default')  % For reproducibility of data and results

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(a)
% Generate and plot the data points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n1 = 50;
n2 = 100;
mu1 = [1; 2];
mu2 = [3; 2];

% Generate dataset (i) 

lambda1 = 1;
lambda2 = 0.25;
theta = 0*pi/6;
[X, Y] = two_2D_Gaussians(n1, n2, mu1, mu2, lambda1, lambda2, theta);

% See below for function two_2D_Gaussians which you need to complete.

% Scatter plot of the generated dataset
X1 = X(:, Y==1);
X2 = X(:, Y==2);

figure(1);subplot(2,2,1);
scatter(X1(1,:),X1(2,:),'o','fill','b');
grid;axis equal;hold on;
xlabel('x_1');ylabel('x_2');
title(['\theta = ',num2str(0),'\times \pi/6']);
scatter(X2(1,:),X2(2,:),'^','fill','r');
axis equal;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code with suitable modifications here to create and plot 
% datasets (ii), (iii), and (iv)
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(b)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% For each phi = 0 to pi in steps of pi/48 compute the signal power, noise 
% power, and snr along direction phi and plot them against phi 

phi_array = 0:pi/48:pi;
signal_power_array = zeros(1,length(phi_array));
noise_power_array = zeros(1,length(phi_array));
snr_array = zeros(1,length(phi_array));
for i=1:1:length(phi_array)
    [signal_power, noise_power, snr] = signal_noise_snr(X, Y, phi_array(i), false);
    % See below for function signal_noise_snr which you need to complete.
    signal_power_array(i) = signal;
    noise_power_array(i) = noise;
    snr_array(i) = snr;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code here to create plots of signal power versus phi, noise
% power versus phi, and snr versus phi and to locate the values of phi
% where the signal power is maximized, the noise power is minimized, and
% the snr is maximized
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For phi = 0, pi/6, and pi/3, generate plots of estimated class 1 and 
% class 2 densities of the projections of the feature vectors along 
% direction phi. To do this, set phi to the desired value, set 
% want_class_density_plots = true; 
% and then invoke the function: 
% signal_noise_snr(X, Y, phi, want_class_density_plots);
% Insert your script here 
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(c)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the LDA solution by writing and invoking a function named LDA 

w_LDA = LDA(X,Y);

% See below for the LDA function which you need to complete.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to create a scatter plot and overlay the LDA vector and the 
% difference between the class means. Use can use Matlab's quiver function 
% to do this.
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 4.3(d)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create CCR vs b plot

X_project = w_LDA' * X;
X_project_sorted = sort(X_project);
b_array = X_project_sorted * (diag(ones(1,n))+ diag(ones(1,n-1),-1)) / 2;
b_array = b_array(1:(n-1));
ccr_array = zeros(1,n-1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exercise: decode what the last 6 lines of code are doing and why
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:1:(n-1)
    ccr_array(i) = compute_ccr(X, Y, w_LDA, b_array(i));
end

% See below for the compute_ccr function which you need to complete.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to plote CCR as a function of b and determine the value of b
% which maximizes the CCR.
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Complete the following 4 functions defined below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [X, Y] = two_2D_Gaussians(n1,n2,mu1,mu2,lambda1,lambda2,theta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function should generate a labeled dataset of 2D data points drawn 
% independently from 2 Gaussian distributions with the same covariance 
% matrix but different mean vectors
%
% Inputs:
%
% n1 = number of class 1 examples
% n2 = number of class 2 examples
% mu1 = 2 by 1 class 1 mean vector
% mu2 = 2 by 1 class 2 mean vector
% theta = orientation of eigenvectors of common 2 by 2 covariance matrix shared by both classes
% lambda1 = first eigenvalue of common 2 by 2 covariance matrix shared by both classes
% lambda2 = second eigenvalue of common 2 by 2 covariance matrix shared by both classes
% 
% Outputs:
%
% X = a 2 by (n1 + n2) matrix with first n1 columns containing class 1
% feature vectors and the last n2 columns containing class 2 feature
% vectors
%
% Y = a 1 by (n1 + n2) matrix with the first n1 values equal to 1 and the 
% last n2 values equal to 2


%%%%%%%%%%%%%%%%%%%%%%
%Insert your code here
%%%%%%%%%%%%%%%%%%%%%%

end

function [signal, noise, snr] = signal_noise_snr(X, Y, phi, want_class_density_plots)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code to project data along direction phi and then comput the
% resulting signal power, noise power, and snr 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ...

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To generate plots of estimated class 1 and class 2 densities of the 
% projections of the feature vectors along direction phi, set:
% want_class_density_plots = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if want_class_density_plots == true
    % Plot density estimates for both classes along chosen direction phi
    figure();
    [pdf1,z1] = ksdensity(X_projected_phi_class1);
    plot(pdf1,z1)
    hold on;
    [pdf2,z2] = ksdensity(X_projected_phi_class2);
    plot(pdf2,z2)
    grid on;
    hold off;
    legend('Class 1', 'Class 2')
    xlabel('projected value')
    ylabel('density estimate')
    title('Estimated class density estimates of data projected along \phi = ? \times \pi/6. Ground-truth \phi = \pi/6')
end

end

function w_LDA = LDA(X, Y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert code to compute and return the LDA solution
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

function ccr = compute_ccr(X, Y, w_LDA, b)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Insert your code here to compute the CCR for the given labeled dataset
% (X,Y) when you classify the feature vectors in X using w_LDA and b
% ...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end