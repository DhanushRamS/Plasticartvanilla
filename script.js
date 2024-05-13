const wrapper = document.querySelector('.wrapper');
const loginLink = document.querySelector('.login-link');
const registerLink = document.querySelector('.register-link');
const btnPopup = document.querySelector('.btnLogin-popup');
const iconClose = document.querySelector('.icon-close');
let locationX;
let locationY;

registerLink.addEventListener('click', () => {
    wrapper.classList.add('active');
});

loginLink.addEventListener('click', () => {
    wrapper.classList.remove('active');
});

btnPopup.addEventListener('click', () => {
    wrapper.classList.add('active-popup');
});

iconClose.addEventListener('click', () => {
    wrapper.classList.remove('active-popup');
});
//-----------------------------------------------------------------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', function () {
    console.log("This is running")
    const loginForm = document.getElementById('loginForm');
    const errorContainer = document.getElementById('errorContainer'); // Assuming you have an element for displaying errors

    loginForm.addEventListener('submit', async function (event) {
        event.preventDefault();

        // Get form data
        const email = document.getElementById('email-login').value;
        const password = document.getElementById('passwd-login').value;

        // Validate form data (you can add more validation logic here)
        if (!email || !password) {
            errorContainer.textContent = 'Please fill in all fields.'; // Display error message
            return;
        }

        try {
            // Send form data to an endpoint (replace with your actual API endpoint)
            const response = await fetch('http://localhost:9999/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email, password }),
            });

            if (response.ok) {
                // Handle successful response (e.g., redirect to another page)
                window.location.href = 'scanner.html';
                console.log(response)
            } else {
                // Handle error response (e.g., display error message)
                errorContainer.textContent = 'Login failed. Please try again.';
            }
        } catch (error) {
            console.error('Error sending form data:', error);
            errorContainer.textContent = 'An error occurred. Please try again later.';
        }
    });
});





document.addEventListener('DOMContentLoaded', function () {
    console.log("This is running")
    const registrationForm = document.getElementById('registrationForm');
    const errorContainer = document.getElementById('regErrorContainer'); // Assuming you have an element for displaying errors

    registrationForm.addEventListener('submit', async function (event) {
        event.preventDefault();

        // Get form data
        const email = document.getElementById('email').value;
        const password = document.getElementById('passwd').value;
        const username = document.getElementById('uname').value;

        // Validate form data (you can add more validation logic here)
        if (!email || !password) {
            errorContainer.textContent = 'Please fill in all fields.'; // Display error message
            return;
        }

        try {
            // Send form data to an endpoint (replace with your actual API endpoint)
            const response = await fetch('http://localhost:9999/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email, password , username}),
            });

            if (response.ok) {
                // Handle successful response (e.g., redirect to another page)
                // window.location.href = 'scanner.html';
                wrapper.classList.add('active');
                console.log(response)
            } else {
                // Handle error response (e.g., display error message)
                errorContainer.textContent = 'Registration failed. Please try again.';
            }
        } catch (error) {
            console.error('Error sending form data:', error);
            errorContainer.textContent = 'An error occurred. Please try again later.';
        }


    });
});


