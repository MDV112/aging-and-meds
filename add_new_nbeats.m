function [] = add_new_nbeats(data_path, output_path, n_beats,group_name,h5_type, win_len, phase)
%% This function simply runs in a cascafe two consecutive function in order
% to add new rr division with different n_beats. It is possible to also add
% in a loop for multiplr n_beats. NOTICE THAT WE DELETE THE rr_win_aging
% folder every time. Possible calls:
% add_new_nbeats('aging', 'rr_win_aging', 25,'aging','rr', 0, 0)
% or
% add_new_nbeats('aging', 'rr_win_aging', [25,50,75],'aging','rr', 0, 0)
    close all
    clc
%%  
    func_pwd = pwd;
    origin_folder = 'C:\Users\smorandv.BM\Documents\PhD\aging and meds github\';
    data_path = [origin_folder data_path '\'];
    output_path = [origin_folder output_path '\'];
    if length(n_beats) == 1    
        if isfolder(output_path)
            rmdir(output_path,'s')
        end
        mkdir(output_path)
        dataset_name = ['nbeats_' num2str(n_beats)];
        rr_win_y(data_path, output_path, n_beats)
        cd(func_pwd)
        convet2hdf(output_path,group_name,dataset_name,h5_type,n_beats, win_len, phase)
        cd(func_pwd)
        clc
    else
        for i = 1:length(n_beats)
            if isfolder(output_path)
                rmdir(output_path,'s')
            end
            mkdir(output_path)
            dataset_name = ['nbeats_' num2str(n_beats(i))];
            rr_win_y(data_path, output_path, n_beats(i))
            cd(func_pwd)
            convet2hdf(output_path,group_name,dataset_name,h5_type,n_beats(i), win_len, phase)
            cd(func_pwd)
            clc
        end
    end
end