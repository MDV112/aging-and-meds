function [] = rr_win_y(data_path, output_path, n_beats)
%% This function diveds the rr to time windows and
% saves the data as rr_win plus label as the last 3 elements and adds the
% window number so label has 4 elements [id; age; med; win_number]
% n_beats should not be more than 1000
% data_path should contain the full rr time series of mice. Can be either:
% 'C:\Users\smorandv.BM\Documents\PhD\aging and meds github\aging' or
% 'C:\Users\smorandv.BM\Documents\PhD\aging and meds github\ac8'
% output_path will contain the segmented rr. Can be either:
% 'C:\Users\smorandv.BM\Documents\PhD\aging and meds github\rr_win_aging\' or
% 'C:\Users\smorandv.BM\Documents\PhD\aging and meds github\rr_win_ac8\'
%%
    close all
    clc
%%
    if not(isfolder('C:\Users\smorandv.BM\Documents\PhD\aging and meds github\rr_win_aging\'))
        mkdir('C:\Users\smorandv.BM\Documents\PhD\aging and meds github\rr_win_aging\')
    end
    cd(data_path)
    listing = dir('*.mat');
%     m = 1e6;
%     for i = 1:length(listing)
%         load(listing(i).name)
%         y = labeled_rr(end-2:end);
%         rr = labeled_rr(1:end-3);
%         if length(rr) < m
%             m = length(rr);
%         end
%     end
    h = 1;
    for i = 1:length(listing)
% after checking n_beats should not be more than 1000 because m resulted in
% 3190
%         if m <= 3*n_beats + 1
%             sprintf('n_beats is too large')
%             break;
%         end
        load(listing(i).name)
        y = labeled_rr(end-2:end);
        rr = labeled_rr(1:end-3);
        win_number = 1;
        for p = 1:n_beats:length(rr)-n_beats
            rr_win = rr(p:p+n_beats-1);
            lbl = [y; win_number];
            labled_win = [rr_win; lbl];
            save([output_path num2str(h) '.mat'], 'labled_win')
            h = h + 1;
            win_number = win_number + 1;
        end
        fprintf('mouse # %d \n',i)
    end
end