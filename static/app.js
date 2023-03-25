var audioTrack = WaveSurfer.create({
  container: ".audio",
  waveColor: "#eee",
  progressColor: "#64dfdf",
  barWidth: 2,
});

audioTrack.load("static/uploadedData/Bus_004.wav");

const playBtn = document.querySelector(".play-btn");
const stopBtn = document.querySelector(".stop-btn");
const muteBtn = document.querySelector(".mute-btn");
const volumeSlider = document.querySelector(".volume-slider");

playBtn.addEventListener("click", () => {
  audioTrack.playPause();

  if (audioTrack.isPlaying()) {
    playBtn.classList.add("playing");
  } else {
    playBtn.classList.remove("playing");
  }
});

stopBtn.addEventListener("click", () => {
  audioTrack.stop();
  playBtn.classList.remove("playing");
});

volumeSlider.addEventListener("mouseup", () => {
  changeVolume(volumeSlider.value);
});

const changeVolume = (volume) => {
  if (volume == 0) {
    muteBtn.classList.add("muted");
  } else {
    muteBtn.classList.remove("muted");
  }
  audioTrack.setVolume(volume);
};

muteBtn.addEventListener("click", () => {
  if (muteBtn.classList.contains("muted")) {
    muteBtn.classList.remove("muted");
    audioTrack.setVolume(0.5);
    volumeSlider.value = 0.5;
  } else {
    muteBtn.classList.add("muted");
    audioTrack.setVolume(0);
    volumeSlider.value = 0;
  }
});

var oReq = new XMLHttpRequest();
function uploadFile(form) {
  var uploadSection = document.getElementsByClassName("upload")[0];
  const formData = new FormData(form);
  var oOutput = document.getElementById("static_file_response");
  oReq.open("POST", "upload_static_file", true);
  oReq.onload = function (oEvent) {
    if (oReq.status == 200) {
      var jsonAnswer = JSON.parse(oReq.response);
      analyzePage()
      addWaveform(jsonAnswer.response, jsonAnswer.name);
      addSpectrogram(jsonAnswer.spectrogram);
      addAnalyzedData(jsonAnswer.categorization, jsonAnswer.clustering, jsonAnswer.similarSounds);
    } else {
      oOutput.innerHTML =
        "Error occurred when trying to upload your file.<br />";
    }
  };
  oOutput.innerHTML = "Audio getting analyzed... Please be patient!";
  var formUploadAudio = document.getElementsByClassName("form-upload-audio")[0];
  formUploadAudio.style.display = "none";
  console.log("Sending file!");
  oReq.send(formData);
}


function addSpectrogram(fileToSpectogram) {
  var spectogramImage = document.getElementById("spectogram")
  spectogramImage.src = fileToSpectogram;
 
}

function addWaveform(filepath, name) {
  var audioContainer = document.getElementsByClassName("audio-container")[0];
  audioContainer.style.display = "block";

  var trackname = document.getElementsByClassName("track-name")[0];
  trackname.innerHTML = name;
  audioTrack.load(filepath);
  
}

function analyzePage(){

  var analyze = document.getElementsByClassName("analyze");
  for (var i = 0; i < analyze.length; i++) {
    analyze[i].style.display = "block";
  }

  var upload = document.getElementById("upload");
  upload.style.display = "none";
}

function addAnalyzedData(categorization, clustering, similarSounds){
  console.log(similarSounds);

  // categorization
  var probability = document.getElementsByClassName("points");
  var classification = document.getElementsByClassName("classification");

  var highestProbability = categorization[0];
  var highestProbabilityIndex = 0;
  for (var i = 0; i < probability.length; i++) {
    probability[i].innerHTML = (categorization[i]*100).toFixed(2).toString() + "%";
    if (categorization[i] > highestProbability){
      highestProbability = categorization[i];
      highestProbabilityIndex = i;
    }
  }

  classification[highestProbabilityIndex].classList.add("selected");

  // clustering

  var scatterplot = document.getElementById("scatterplot");
  scatterplot.src = clustering;

  // similar sounds
  var similarSoundsAudio = document.getElementsByClassName("similar-sound-audio");
  var similarSoundNames = document.getElementsByClassName("similarSoundName");
  var similarSoundProbability = document.getElementsByClassName("similarSoundProbability");
  console.log(similarSoundsAudio[0]);
  console.log(similarSounds[0]);
  for (var i = 0; i < similarSounds.length; i++) {
    console.log(similarSounds[i]);
    similarSoundNames[i].innerHTML = similarSounds[i][0].split("/")[3];
    similarSoundsAudio[i].src = similarSounds[i][0];
    similarSoundProbability[i].innerHTML = (similarSounds[i][1]*100).toFixed(2).toString() + "%";
  }

  var possibility= document.getElementById("possibility");
  possibility.innerHTML = similarSounds;
}