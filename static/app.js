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
      //oOutput.innerHTML = "Uploaded!";
      var jsonAnswer = JSON.parse(oReq.response);
      //analyzeFile(jsonAnswer.response);

      analyzePage()
      addWaveform(jsonAnswer.response, jsonAnswer.name);
      addSpectrogram(jsonAnswer.spectrogram);
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
  // spectogramImage.style.display = "block";
  spectogramImage.src = fileToSpectogram;
  // spectogramImage.srcset = fileToSpectogram + " 1x";
  // spectogramImage.type = "image/png";
  // spectogramImage.target = "_blank";
  // spectogramImage.setAttribute("download", "false");
}

function addWaveform(filepath, name) {
  var audioContainer = document.getElementsByClassName("audio-container")[0];
  audioContainer.style.display = "block";

  var trackname = document.getElementsByClassName("track-name")[0];
  trackname.innerHTML = name;
  audioTrack.load(filepath);
  // audioTrack.play();
  
}

function analyzePage(){

  var analyze = document.getElementsByClassName("analyze");
  for (var i = 0; i < analyze.length; i++) {
    analyze[i].style.display = "block";
  }

  // var introduction = document.getElementById("introduction");
  // introduction.style.display = "none";

  var upload = document.getElementById("upload");
  upload.style.display = "none";
}