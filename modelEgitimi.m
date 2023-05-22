Egitim_Dataseti = imageDatastore('Dataset/Training', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
Dogrulama_Dataseti = imageDatastore('Dataset/Validation', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%CNN modelindeki GoogleNet modelinin bir ornegini olusturmak:
net = googlenet;
analyzeNetwork(net);

%Giris_Katmani = net.Layers(1);
Ozellik_Katmani = net.Layers(142);
Cikis_Siniflandirici = net.Layers(144);

Giris_Katmani_Boyutu = Giris_Katmani.InputSize;

Katman_Grafi = layerGraph(net);

%GirisBoyutu = [512 512 3];
Siniflarin_Sayisi = numel(categories(Egitim_Dataseti.Labels));

%Yeni_Giris_Katmani = imageInputLayer(GirisBoyutu, 'Name', '1. Giris Katmani');
Yeni_Ozellik_Katmani = fullyConnectedLayer(Siniflarin_Sayisi, ...
    'Name', '142. Ozellik Katmani', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);
Yeni_Siniflandirma_Katman = classificationLayer('Name', '144. Son katman');

%Katman_Grafi = replaceLayer(Katman_Grafi, Giris_Katmani.Name, Yeni_Giris_Katmani);
Katman_Grafi = replaceLayer(Katman_Grafi, Ozellik_Katmani.Name, Yeni_Ozellik_Katmani);
Katman_Grafi = replaceLayer(Katman_Grafi, Cikis_Siniflandirici.Name, Yeni_Siniflandirma_Katman);

analyzeNetwork(Katman_Grafi)

Resized_Egitim_Dataseti = augmentedImageDatastore([224 224], Egitim_Dataseti);
Resized_Dogrulama_Dataseti = augmentedImageDatastore([224 224], Dogrulama_Dataseti);

ItirasyondakiFoto = 250;
Dogrulama_frekansi = floor(numel(Resized_Egitim_Dataseti.Files)/ItirasyondakiFoto);
Egitim_ayarlari = trainingOptions('sgdm',...
    'MiniBatchSize', ItirasyondakiFoto, ...
    'MaxEpochs', 6,...
    'InitialLearnRate', 3e-4,...
    'Shuffle', 'every-epoch', ...
    'ValidationData', Resized_Dogrulama_Dataseti, ...
    'ValidationFrequency', Dogrulama_frekansi, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(Resized_Egitim_Dataseti, Katman_Grafi, Egitim_ayarlari);
