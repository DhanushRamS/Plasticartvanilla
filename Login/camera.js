// Bring elements, create setting etc

var video = document.getElementById("video");
var cameraShot = document.getElementById("camera-snapshot");
var canvasShot = document.getElementById(".canvas");

// Get access to the camera
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({ video: true }).then(function (stream) {
    video.srcObject = stream;
    video.play();
  });
}

// Create functionality
// cameraShot.addEventListener('click', mouseOvershot);

// function mouseOvershot(e) {
//     console.log(e.target)
//     Array.from(canvasShot).forEach(function(shot) {
//         console.log(shot)
//     })
// }

// Trigger photo take
function dataURItoBlob(dataURI) {
  // convert base64/URLEncoded data component to raw binary data held in a string
  var byteString;
  if (dataURI.split(",")[0].indexOf("base64") >= 0)
    byteString = atob(dataURI.split(",")[1]);
  else byteString = unescape(dataURI.split(",")[1]);

  // separate out the mime component
  var mimeString = dataURI.split(",")[0].split(":")[1].split(";")[0];

  // write the bytes of the string to a typed array
  var ia = new Uint8Array(byteString.length);
  for (var i = 0; i < byteString.length; i++) {
    ia[i] = byteString.charCodeAt(i);
  }
  return new Blob([ia], { type: mimeString });
}

// document.getElementById('snap').addEventListener('click', function() {
//     var labeledImage = document.createElement('div');
//     labeledImage.className = 'labeledImage'
//     var canvas = document.createElement('canvas');
//     var context = canvas.getContext('2d')

//     canvas.width = '280'
//     canvas.height = '200'
//     canvas.className = 'canvas'
//     canvas.style.margin = '0px 1.5px'
//     console.log(canvas)

//     labeledImage.appendChild(canvas);
//     context.drawImage(video, 0, 0, 280, 200)

//     var imageData = canvas.toDataURL('image/jpeg');
//     imageData = dataURItoBlob(imageData)
//     const formData = new FormData();

//     formData.append('image', imageData);
//     // Send the image data to the Python backend
//     fetch('http://127.0.0.1:5000/upload', {
//         method: 'POST',
//         body: formData
//     })
//     .then(response => response.text())
//     .then(responseText => {
//         // Display the response below the image
//         var responseElement = document.createElement('p');
//         responseElement.textContent = 'Server Response: ' + responseText;
//         labeledImage.appendChild(responseElement);
//         cameraShot.appendChild(labeledImage);
//     })
//     .catch(error => console.error('Error sending image:', error));
// })

document.getElementById("snap").addEventListener("click", function () {
  var labeledImage = document.createElement("div");
  labeledImage.className = "labeledImage";
  var canvas = document.createElement("canvas");
  var context = canvas.getContext("2d");

  canvas.width = "280";
  canvas.height = "200";
  canvas.className = "canvas";
  canvas.style.margin = "0px 1.5px";
  console.log(canvas);

  labeledImage.appendChild(canvas);
  context.drawImage(video, 0, 0, 280, 200);

  var imageData = canvas.toDataURL("image/jpeg");
  imageData = dataURItoBlob(imageData);
  const formData = new FormData();

  formData.append("image", imageData);
  // Send the image data to the Python backend
  fetch("http://127.0.0.1:5000/upload", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.text())
    .then((responseText) => {
      // Display the response below the image

      var responseElement = document.createElement("p");
      responseElement.textContent =
        "The Object in the picture is: " + responseText;
      labeledImage.appendChild(responseElement);
      cameraShot.appendChild(labeledImage);
      return responseText;
    })
    .then((trashType) => {
      const type = trashType;
      console.log(type);
      if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition((position) => {
          const lat = position.coords.latitude;
          const long = position.coords.longitude;
          fetch("http://localhost:9999/send_trash", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ lat, long, type }),
          }).then((res) => {
            console.log("Trash Sent");
          });
        });
      } else {
        throw "Geolocation is not supported by this browser.";
      }
    })
    .catch((error) => console.error("Error sending image:", error));
});
