const wrapper = document.querySelector(".wrapper");
const loginLink = document.querySelector(".login-link");
const registerLink = document.querySelector(".register-link");
const btnPopup = document.querySelector(".btnLogin-popup");
const iconClose = document.querySelector(".icon-close");

registerLink.addEventListener("click", () => {
  wrapper.classList.add("active");
});

loginLink.addEventListener("click", () => {
  wrapper.classList.remove("active");
});

btnPopup.addEventListener("click", () => {
  wrapper.classList.add("active-popup");
});

iconClose.addEventListener("click", () => {
  wrapper.classList.remove("active-popup");
});

function getLocation() {}

//-----------------------------------------------------------------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", function () {
  console.log("This is running -- vendor");
  const loginForm = document.getElementById("loginForm");
  const errorContainer = document.getElementById("errorContainer"); // Assuming you have an element for displaying errors

  loginForm.addEventListener("submit", async function (event) {
    event.preventDefault();

    // Get form data
    const email = document.getElementById("email-login").value;
    const password = document.getElementById("passwd-login").value;

    // Validate form data (you can add more validation logic here)
    if (!email || !password) {
      errorContainer.textContent = "Please fill in all fields."; // Display error message
      return;
    }

    try {
      // Send form data to an endpoint (replace with your actual API endpoint)
      const response = await fetch("http://localhost:9999/vendor/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email, password }),
      });

      if (response.ok) {
        // Handle successful response (e.g., redirect to another page)
        window.location.href = "./vendor.html";
        console.log(response);
      } else {
        // Handle error response (e.g., display error message)
        errorContainer.textContent = "Login failed. Please try again.";
      }
    } catch (error) {
      console.error("Error sending form data:", error);
      errorContainer.textContent = "An error occurred. Please try again later.";
    }
  });
});

document.addEventListener("DOMContentLoaded", function () {
  console.log("This is running");
  const registrationForm = document.getElementById("registrationForm");
  const errorContainer = document.getElementById("regErrorContainer"); // Assuming you have an element for displaying errors

  registrationForm.addEventListener("submit", async function (event) {
    event.preventDefault();

    // Get form data
    const email = document.getElementById("email").value;
    const password = document.getElementById("passwd").value;
    const username = document.getElementById("uname").value;

    // Validate form data (you can add more validation logic here)
    if (!email || !password) {
      errorContainer.textContent = "Please fill in all fields."; // Display error message
      return;
    }

    try {
      if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(async (position) => {
          const locationX = position.coords.latitude;
          const locationY = position.coords.longitude;
          console.log(locationX);
          console.log(locationY);
          console.log(
            JSON.stringify({
              email,
              password,
              username,
              lat: locationX,
              long: locationY,
            })
          );
          const response = await fetch(
            "http://localhost:9999/vendor/register",
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                email,
                password,
                username,
                lat: locationX,
                long: locationY,
              }),
            }
          );
          if (response.ok) {
            // Handle successful response (e.g., redirect to another page)
            // window.location.href = 'scanner.html';
            wrapper.classList.add("active");
            console.log(response);
          } else {
            // Handle error response (e.g., display error message)
            errorContainer.textContent =
              "Registration failed. Please try again.";
          }
        });
      } else {
        console.log("Geolocation is not supported by this browser.");
      }
      // Send form data to an endpoint (replace with your actual API endpoint)
    } catch (error) {
      console.error("Error sending form data:", error);
      errorContainer.textContent = "An error occurred. Please try again later.";
    }
  });
});

document.addEventListener("DOMContentLoaded", function () {
  const aboutLink = document.querySelector('a[href="#about"]');

  aboutLink.addEventListener("click", function (event) {
    event.preventDefault();

    const targetPosition = document.body.scrollHeight;

    window.scrollTo({
      top: targetPosition,
      //   behavior: "smooth",
    });
  });
});
