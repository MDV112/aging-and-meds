function [] = hrv_processing(data_type,win_len,phase)
%% This function saves the hrv features to h5 after being processed by mhrv
    old_path = pwd;
    switch data_type
        case 'aging'
            count = 0;
            for age = 6:3:30
                rec_dir = ['C:\Users\smorandv.STAFF\Documents\PhD\Aging Data\C57 '...
                    num2str(age) 'm mat'];
                cd(rec_dir)
                load('both_batchs.mat')
                bas_hrv = batch_data_bas.hrv_tables('ALL');
                abk_hrv = batch_data_abk.hrv_tables('ALL');
                count = count + size(bas_hrv,1) + size(abk_hrv,1);
            end
            h5create('C:\Users\smorandv.STAFF\Documents\PhD\aging and meds\hrv_mse.h5',['/aging/win_len_' num2str(win_len) '_phase_' num2str(phase) '_input'],[count size(abk_hrv,2)])
            h5create('C:\Users\smorandv.STAFF\Documents\PhD\aging and meds\hrv_mse.h5',['/aging/win_len_' num2str(win_len) '_phase_' num2str(phase) '_label'],[count 4])
            hrv_name = ['win_len_' num2str(win_len) '_phase_' num2str(phase) '_input'];
            y_name = ['win_len_' num2str(win_len) '_phase_' num2str(phase) '_label'];
            h = 1;
            for age = 6:3:30
                rec_dir = ['C:\Users\smorandv.STAFF\Documents\PhD\Aging Data\C57 '...
                    num2str(age) 'm mat'];
                cd(rec_dir)
                load('both_batchs.mat')
                bas_hrv = batch_data_bas.hrv_tables('ALL');
                bas_vars = bas_hrv.Properties.VariableNames;
                bas_rows = bas_hrv.Properties.RowNames;
                med = 0;
                bas_mat = table2array(bas_hrv);
                for j = 1:size(bas_rows,1)
                    n = bas_rows{j};
                    idx1 = strfind(n, '(');
                    idx2 = strfind(n, ')');
                    id = str2double(n(1 + idx1:idx2 - 1));
                    undscr = strfind(n, '_');
                    idx = undscr(end);
                    win_num = str2double(n(idx+1:end));
                    hrv_feat = bas_mat(j,:);
                    y = [id age med win_num];
                    h5write('C:\Users\smorandv.STAFF\Documents\PhD\aging and meds\hrv_mse.h5',['/aging/' hrv_name],hrv_feat, [h 1],size(hrv_feat))
                    h5write('C:\Users\smorandv.STAFF\Documents\PhD\aging and meds\hrv_mse.h5',['/aging/' y_name],y, [h 1],size(y))
                    fprintf('saved %d/%d to C:\\Users\\smorandv.STAFF\\Documents\\PhD\\aging and meds\\hrv_mse.h5 \n',h,count)
                    h = h + 1;
                end

                abk_hrv = batch_data_abk.hrv_tables('ALL');
                abk_vars = abk_hrv.Properties.VariableNames;
                abk_rows = abk_hrv.Properties.RowNames;
                med = 1;
                abk_mat = table2array(abk_hrv);
                for j = 1:size(abk_rows,1)
                    n = abk_rows{j};
                    idx1 = strfind(n, '(');
                    idx2 = strfind(n, ')');
                    id = str2double(n(1 + idx1:idx2 - 1));
                    undscr = strfind(n, '_');
                    idx = undscr(end);
                    win_num = str2double(n(idx+1:end));
                    hrv_feat = abk_mat(j,:);
                    y = [id age med win_num];
                    h5write('C:\Users\smorandv.STAFF\Documents\PhD\aging and meds\hrv_mse.h5',['/aging/' hrv_name],hrv_feat, [h 1],size(hrv_feat))
                    h5write('C:\Users\smorandv.STAFF\Documents\PhD\aging and meds\hrv_mse.h5',['/aging/' y_name],y, [h 1],size(y))
                    fprintf('saved %d/%d to C:\\Users\\smorandv.STAFF\\Documents\\PhD\\aging and meds\\hrv_mse.h5 \n',h,count)
                    h = h + 1;
                end
            end
            save(['aging_win_len_' num2str(win_len) '_phase_' num2str(phase) '_featuresNames.mat'],'bas_vars')
        case 'ac8'
    end
    cd(old_path)
    
end