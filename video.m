load('trained_net.mat');
yuzalgilayici = vision.CascadeObjectDetector();
videokuyucu = vision.VideoFileReader('C:\Users\user\Desktop\_\video1.mp4');

% Create a video player to display the processed frames
videobilgileri = info(videokuyucu);
videogoruntuleyici = vision.VideoPlayer('Position', [300 300 videobilgileri.VideoSize + 30]);

while ~isDone(videokuyucu)
    videokareleri = step(videokuyucu); 

    cep = step(yuzalgilayici, videokareleri);
    
    for i = 1:size(cep, 1)
        yuz = imcrop(videokareleri, cep(i, :));
        G = imresize(yuz, [224, 224]);
        [Cinsiyet, Yuzdelik] = classify(net, G);

        cikti = insertObjectAnnotation(videokareleri, 'rectangle', cep(i, :), '');
    end

    step(videogoruntuleyici, cikti);
end

release(videokuyucu);
release(videogoruntuleyici);
