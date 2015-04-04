
%float
function test_stats
	codebook_file = '/net/per610a/export/das11f/plsang/trecvidmed13/feature/codebook/mfcc.bg.rastamat/data/codebook.gmm.256.39.mat';
	load(codebook_file);
	
	fisher_params.grad_weights = false;		% "soft" BOW
    fisher_params.grad_means = true;		% 1st order
    fisher_params.grad_variances = true;	% 2nd order
    fisher_params.alpha = single(1.0);		% power normalization (set to 1 to disable)
    fisher_params.pnorm = single(0.0);		% norm regularisation (set to 0 to disable)
	
    feats = single(rand(39,50000));
    
    num_seg = 1000; 
    
    cpp_handle = mexFisherEncodeHelperSP('init', codebook, fisher_params);
	 
    tic;
	code1 = mexFisherEncodeHelperSP('encode', cpp_handle, feats);
    fprintf('Encoding without saving stats, time = %f \n', toc);
    
    tic;
    [code2, stats_] = mexFisherEncodeHelperSP('encodestats', cpp_handle, feats);
    fprintf('Encoding with saving stats, time = %f \n', toc);
    
    range = 1:num_seg:size(feats, 2);
    
    stats = zeros(1 + 256 + 2*256*size(feats, 1), length(range));
    tic;
    for ii=1:length(range),
        start_idx = range(ii);
        end_idx = start_idx + num_seg - 1;
        if end_idx > size(feats, 2),
            end_idx = size(feats, 2);
        end
        [~, stats(:, ii)] = mexFisherEncodeHelperSP('encodestats', cpp_handle, feats(:,start_idx:end_idx));
    end
    stats = single(sum(stats, 2));
    
    fprintf('Encoding & savign stats with %d segs, time = %f \n', length(range), toc);
    
    tic;
    code3 = mexFisherEncodeHelperSP('getfkstats', cpp_handle, stats);
    fprintf('Encoding from stats, time = %f \n', toc);
    
    %mexFisherEncodeHelperSP('statstest', cpp_handle, feats, stats);
    
    fprintf('stats diff %f \n', sum(stats - stats_));
    fprintf('code diff %f \n', sum(code1 - code3));
    
    mexFisherEncodeHelperSP('clear', cpp_handle);
	
    if sum(~eq(code1, code2)) == 0,
		fprintf('1-2 Equal!');
	end
    
    if sum(~eq(code1, code3)) == 0,
		fprintf('1-3 Equal!');
	end
    
end

function test_accumulate_1
	codebook_file = '/net/per900a/raid0/plsang/trecvidmed10/feature/bow.codebook.trecvidmed10.devel/densetrajectory.mbh/data/codebook.gmm.256.mat';
	load(codebook_file);
	
	fisher_params.grad_weights = false;		% "soft" BOW
    fisher_params.grad_means = true;		% 1st order
    fisher_params.grad_variances = true;	% 2nd order
    fisher_params.alpha = single(1.0);		% power normalization (set to 1 to disable)
    fisher_params.pnorm = single(0.0);		% norm regularisation (set to 0 to disable)
	
    feats = single(rand(192, 2000));
    
    cpp_handle = mexFisherEncodeHelperSP('init', codebook, fisher_params);
	 
	code1 = mexFisherEncodeHelperSP('encode', cpp_handle, feats);
    
    mexFisherEncodeHelperSP('clear', cpp_handle);
    
    cpp_handle = mexFisherEncodeHelperSP('init', codebook, fisher_params);
	 
	mexFisherEncodeHelperSP('accumulate', cpp_handle, feats(:,1:1000));
    
    mexFisherEncodeHelperSP('accumulate', cpp_handle, feats(:,1001:2000));
    
    code2 = mexFisherEncodeHelperSP('getfk', cpp_handle);
    
    if sum(~eq(code1, code2)) == 0,
		fprintf('Equal!');
	end
    
end