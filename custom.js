// document.addEventListener('DOMContentLoaded', function () {
//     const wrapper = document.querySelector('.wrapper');
//     const loginLink = document.querySelector('.login-link');
//     const btnPopup = document.querySelector('.btnLogin-popup');
//     const iconClose = document.querySelector('.icon-close');

//     loginLink.addEventListener('click', () => {
//         wrapper.classList.remove('active-popup');
//     });

//     btnPopup.addEventListener('click', () => {
//         wrapper.classList.add('active-popup');
//     });

//     iconClose.addEventListener('click', () => {
//         wrapper.classList.remove('active-popup');
//     });

//     const loginForm = document.getElementById('loginForm');
//     const registerForm = document.getElementById('registerForm');
//     const errorContainer = document.getElementById('errorContainer');
//     const registerErrorContainer = document.getElementById('registerErrorContainer');

//     loginForm.addEventListener('submit', async function (event) {
//         event.preventDefault();

//         const email = document.getElementById('email-login').value;
//         const password = document.getElementById('passwd-login').value;

//         if (!email || !password) {
//             errorContainer.textContent = 'Please fill in all fields.';
//             return;
//         }

//         try {
//             const response = await fetch('http://localhost:9999/login', {
//                 method: 'POST',
//                 headers: {
//                     'Content-Type': 'application/json',
//                 },
//                 body: JSON.stringify({ email, password }),
//             });

//             if (response.ok) {
//                 console.log(response);
//                 // Handle successful login
//             } else {
//                 errorContainer.textContent = 'Login failed. Please try again.';
//             }
//         } catch (error) {
//             console.error('Error sending login data:', error);
//             errorContainer.textContent = 'An error occurred. Please try again later.';
//         }

//         loginForm.reset();
//     });

//     registerForm.addEventListener('submit', async function (event) {
//         event.preventDefault();

//         const username = document.getElementById('username-register').value;
//         const email = document.getElementById('email-register').value;
//         const password = document.getElementById('passwd-register').value;

//         if (!username || !email || !password) {
//             registerErrorContainer.textContent = 'Please fill in all fields.';
//             return;
//         }

//         try {
//             const response = await fetch('http://localhost:9999/register', {
//                 method: 'POST',
//                 headers: {
//                     'Content-Type': 'application/json',
//                 },
//                 body: JSON.stringify({ username, email, password }),
//             });

//             if (response.ok) {
//                 console.log(response);
//                 // Handle successful registration
//             } else {
//                 registerErrorContainer.textContent = 'Registration failed. Please try again.';
//             }
//         } catch (error) {
//             console.error('Error sending registration data:', error);
//             registerErrorContainer.textContent = 'An error occurred. Please try again later.';
//         }

//         registerForm.reset();
//     });
// });
