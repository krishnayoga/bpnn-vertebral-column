%% Backpropagation with Fisher Iris data input

%%%%%%%%%% INISIALISASI DATA %%%%%%%%%%
% Inisialisasi data
file = 'test_data.xlsx';
mat_in = xlsread(file,1);
mat_target = xlsread(file,2);

[length_in_row,length_in_col] = size(mat_in);
[length_out_row,length_out_col] = size(mat_target);

hidden_n = 5;
error_target = 0.1;
error = Inf;
epoch = 1;
alpha = 0.4;
baris = 1;
stop = 0;
max_epoch = 500000;
miu = 0.9;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% NORMALISASI DATA %%%%%%%%%%%%%%

% Data input
for m = 1 : length_in_row
    for n = 1 : length_in_col
       mat_in(m,n) = ((mat_in(m,n) - min(mat_in(:,n)))/(max(mat_in(:,n)) - min(mat_in(:,n))));
    end
end

% Data target
%for m = 1 : length_out_row
%    for n = 1 : length_out_col
%        mat_target(m,n) = ((mat_target(m,n) - min(mat_target(:,n)))/(max(mat_target(:,n)) - min(mat_target(:,n))));
%    end
%end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% INISIALISASI BOBOT DENGAN NGUYEN WIDROW %%%%%%%%%%

%beta
beta=0.7*(hidden_n).^(1/length_in_row); 

%set bobot v
weight_hidden_in=[rand(length_in_col,hidden_n)-0.5];

%hitung vij
weight_hidden_in_abs = sqrt(sum(sum(weight_hidden_in.^2)));
%update bobot
for i = 1:length_in_col
    for j = 1:hidden_n
        weight_hidden_in(i,j) = beta.*weight_hidden_in(i,j).*(1./weight_hidden_in_abs);
    end
end

%set bobot w
weight_hidden_out=[rand(hidden_n,length_out_col)-0.5];

%hitung wij
weight_hidden_out_abs = sqrt(sum(sum(weight_hidden_out.^2)));

%update bobot
for i = 1:hidden_n
    for j = 1:length_out_col
        weight_hidden_out(i,j) = beta.*weight_hidden_out(i,j).*(1./weight_hidden_out_abs);
    end
end

% Inisialisasi bias
bias_hidden_in = [rand(1,hidden_n)-beta];
bias_hidden_out = [rand(1,length_out_col)-beta];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% INISIALISASI VEKTOR DELTA %%%%%%%%%%

%inisialisasi matrix delta
%delta bobot v&w
delta_hidden_in = zeros(length_in_col,hidden_n);
delta_hidden_out = zeros(hidden_n,length_out_col);
%delta bobot bias 
delta_bias_hidden_in = zeros(1,hidden_n);
delta_bias_hidden_out = zeros(1,length_out_col);
%bobot momentum lama
delta_hidden_out_old = zeros(hidden_n,length_out_col);
delta_hidden_in_old = zeros(length_in_col,hidden_n);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% MULAI TRAINING %%%%%%%%%%

while stop == 0 && epoch ~= max_epoch
    
    %step 3
    input = mat_in(baris,:);
    target = mat_target(baris,:);
    
    %step 4
    z_in = bias_hidden_in + input * weight_hidden_in;
    
    for j=1:hidden_n
        z_aktiv(1,j) = 1/(1+exp(-z_in(1,j)));
    end
    
    %step 5
    y_in = bias_hidden_out + z_aktiv * weight_hidden_out;
    
    for k=1:length_out_col
        y_aktiv(1,k) = 1 /( 1 + exp( -y_in(1,k) ) );
    end
    
    momentum_wjk = miu * delta_hidden_out_old;
    momentum_vij = miu * delta_hidden_in_old;
    
    %step 6
    for k=1:length_out_col
        do_k(1,k) = (target(1,k) - y_aktiv(1,k)) * (exp(-y_in(k))/((1 +exp(-y_in(k))).^2 ));
    end
    
    delta_hidden_out = alpha .* z_aktiv' * do_k + momentum_wjk;
    delta_bias_hidden_out = alpha .* do_k;
    
    %step 7
    do_in = do_k * weight_hidden_out';
  
    for j=1:hidden_n
        do_j(1,j) = (do_in(1,j)) .* (exp(-z_in(j))./((1 +exp(-z_in(j))).^2 ));
    end
    
    delta_hidden_in = (alpha .* do_j' * input)' + momentum_vij;
    delta_bias_hidden_in = alpha .* do_j;
    
    %step 8
    weight_hidden_out = weight_hidden_out + delta_hidden_out;
    bias_hidden_out = bias_hidden_out + delta_bias_hidden_out;
    
    weight_hidden_in = weight_hidden_in + delta_hidden_in;
    bias_hidden_in = bias_hidden_in + delta_bias_hidden_in;
    
    %step 9
    error(1,baris) = 0.5 .* sum((target-y_aktiv).^2);
    
    if baris == length_in_row
        error_tot(1,epoch) = sum(error)/length_in_row;
        if error_tot(1,epoch) < error_target
            stop = 1;
        end
        baris = 0;
        epoch = epoch+1;
    end
    
    delta_hidden_out_old = delta_hidden_out;
    delta_hidden_in_old = delta_hidden_in;
    
    baris = baris+1;
    
end

xlswrite(file,weight_hidden_in,5);
xlswrite(file,weight_hidden_out,6);
xlswrite(file,bias_hidden_in,7);
xlswrite(file,bias_hidden_out,8);
plot(error_tot)

