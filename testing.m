%% Backpropagation with Fisher Iris data input

%%%%%%%%%% INISIALISASI DATA %%%%%%%%%%
% Inisialisasi data
file = 'test_data.xlsx';
mat_in = xlsread(file,3);
mat_target = xlsread(file,4);
weight_hidden_in = xlsread(file,5);
weight_hidden_out = xlsread(file,6);
bias_hidden_in = xlsread(file,7);
bias_hidden_out = xlsread(file,8);
hidden_n = 5;
benar = 0;
salah = 0;

[length_in_row,length_in_col] = size(mat_in);
[length_out_row,length_out_col] = size(mat_target);

baris = 1;
stop = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% NORMALISASI DATA %%%%%%%%%%%%%%

% Data input
for m = 1 : length_in_row
    for n = 1 : length_in_col
       mat_in(m,n) = ((mat_in(m,n) - min(mat_in(:,n)))/(max(mat_in(:,n)) - min(mat_in(:,n))));
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% MULAI TRAINING %%%%%%%%%%

while stop == 0
    
    %step 3
    input = mat_in(baris,:);
    target = mat_target(baris,:);
    
    %step 4
    xi_vij = input * weight_hidden_in;
    z_in = bias_hidden_in + xi_vij;
    
    for j=1:hidden_n
        z_aktiv(1,j) = 1/(1+exp(-z_in(1,j)));
    end
    
    %step 5
    zj_wjk = z_aktiv * weight_hidden_out;
    y_in = bias_hidden_out + zj_wjk;
    
    for k=1:length_out_col
        y_aktiv(1,k) = 1 /( 1 + exp( -y_in(1,k) ) );
    end
    
    for i=1:length_out_col
        if y_aktiv(1,i) == max(y_aktiv)
           y_aktiv(1,i) = max(target);
        else
            y_aktiv(1,i) = 0;
        end
    end
    
    error = target - y_aktiv;
    
    if error == zeros(1,length_out_col)
        benar = benar + 1;
    else error ~= zeros(1,length_out_col)
        salah = salah+1;
    end
    
    if baris == length_in_row
        stop = 1;
    end
    
    baris = baris+1;
end

rate = benar/length_in_row;