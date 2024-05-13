//For Traversal from main to indexs-----------------------------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', function () {
    const user = document.getElementById('login user');

    loginForm.addEventListener('submit', function (event) {
        event.preventDefault();
        window.location.href = 'user_index.html';
    });
});

document.addEventListener('DOMContentLoaded', function () {
    const user = document.getElementById('login vendor');

    loginForm.addEventListener('submit', function (event) {
        event.preventDefault();
        window.location.href = 'vendor_index.html';
    });
});

document.addEventListener('DOMContentLoaded', function () {
    const user = document.getElementById('login admin');

    loginForm.addEventListener('submit', function (event) {
        event.preventDefault();
        window.location.href = 'admin_index.html';
    });
});