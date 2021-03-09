function [] = convet2hdf(data_path,group_name,dataset_name,h5_type,nbeats, win_len, phase)
%% This function stores data within h5 file
% The data is saved as rows in rr because h5py reads it as a transpose
% data_path can be for instance rr_win
% NOTICE: due to use of the function dir, we cannot assume ascending order
% of age in the h5
% group_name can be either one of the switch
% dataset_name can be one of the choices according to tasks document
% h5_type can be either 'rr' or 'hrv'
% nbeats is used for rr

    
    cd(data_path)
    listing = dir('*.mat');
    if strcmp(h5_type,'rr')
        switch group_name
            case 'aging'
                h5create('C:\Users\smorandv.STAFF\Documents\PhD\aging and meds\rr.h5',['/aging/nbeats_' double2str(nbeats) '_input'],[length(listing) nbeats])
                h5create('C:\Users\smorandv.STAFF\Documents\PhD\aging and meds\rr.h5',['/aging/nbeats_' double2str(nbeats) '_label'],[length(listing) 4])
                for i = 1:length(listing)
                    load(listing(i).name)
                    rr = labled_win(1:end-4);
                    rr = rr';
                    y = labled_win(end-3:end); % [id; age; med; win_number]
                    y = y';
                    rr_name = [dataset_name '_input'];
                    y_name = [dataset_name '_label'];
                    h5write('C:\Users\smorandv.STAFF\Documents\PhD\aging and meds\rr.h5',['/aging/' rr_name],rr, [i 1],size(rr))
                    h5write('C:\Users\smorandv.STAFF\Documents\PhD\aging and meds\rr.h5',['/aging/' y_name],y, [i 1],size(y))
                    fprintf('saved %d/%d to rr.h5 \n',i,length(listing))
                end    
            case 'ac8'
        end
    else % hrv
        switch group_name
        case 'aging' % take care of the size ,[length(listing) hrv features]
            h5create('C:\Users\smorandv.STAFF\Documents\PhD\aging and meds\hrv.h5',['/aging/win_len_' double2str(win_len) '_phase_' double2str(phase) '_input'])
            h5create('C:\Users\smorandv.STAFF\Documents\PhD\aging and meds\hrv.h5',['/aging/win_len_' double2str(win_len) '_phase_' double2str(phase) '_label'],[length(listing) 4])
            for i = 1:length(listing)
                load(listing(i).name)
%                 rr = labled_win(1:end-3);
%                 rr = rr';
%                 y = labled_win(end-2:end);
%                 y = y';
                hrv_name = [dataset_name '_input'];
                y_name = [dataset_name '_label'];
                h5write('C:\Users\smorandv.STAFF\Documents\PhD\aging and meds\hrv.h5',['/aging/' hrv_name],rr, [i 1],size(rr))
                h5write('C:\Users\smorandv.STAFF\Documents\PhD\aging and meds\hrv.h5',['/aging/' y_name],y, [i 1],size(y))
                fprintf('saved %d/%d to rr.h5 \n',i,length(listing))
            end    
        case 'ac8'
        end
    end
end    
    