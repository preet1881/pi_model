% -------------------------------------------------------------------------
% function imdb = getCifarImdb(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted


dataDir               = pwd;
contrastNormalization = 1;
whitenData            = 1;
unpackPath            = fullfile(dataDir, 'cifar-10-batches-mat');
files                 = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, dataDir) ;
end

data   = cell(1, numel(files));
labels = cell(1, numel(files));
sets   = cell(1, numel(files));
for fi = 1:numel(files)
  fd         = load(files{fi}) ;
  data{fi}   = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.labels' + 1; % Index from 1
  sets{fi}   = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));

% remove mean in any case
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

if contrastNormalization
  z = reshape(data,[],60000) ;
  z = bsxfun(@minus, z, mean(z,1)) ;
  n = std(z,0,1) ;
  z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
  data = reshape(z, 32, 32, 3, []) ;
end

if whitenData
  z = reshape(data,[],60000) ;
  W = z(:,set == 1)*z(:,set == 1)'/60000 ;
  [V,D] = eig(W) ; % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  data = reshape(z, 32, 32, 3, []) ;
end

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));
if contrastNormalization && whitenData
    save_file = 'cifar10_constrast_ZCA.mat';
elseif contrastNormalization
    save_file = 'cifar10_constrast.mat';
elseif whitenData
    save_file = 'cifar10_ZCA.mat';
else    
    save_file ='cifar10.mat'
end

data    = permute(data, [4, 3, 1, 2]);
labels  = single(cat(2, labels{:})) ;
classes = clNames.label_names;
save(save_file, 'data', 'labels', 'classes', 'set')