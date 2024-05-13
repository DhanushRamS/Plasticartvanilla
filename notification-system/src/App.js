// src/App.js
import React, { useState } from 'react';
import NotificationList from './components/NotificationList';
import NotificationForm from './components/NotificationForm';

const App = () => {
    const [notifications, setNotifications] = useState([]);

    const handleSendNotification = (message) => {
        // Add your logic to send notifications to the server
        // Update the 'notifications' state accordingly
        // Example: setNotifications([...notifications, { id: Date.now(), message, read: false }]);
    };

    return (
        <div className="app">
            <h1>Notification System</h1>
            <NotificationList notifications={notifications} />
            <NotificationForm onSendNotification={handleSendNotification} />
        </div>
    );
};

export default App;
