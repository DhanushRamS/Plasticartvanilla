// src/components/NotificationForm.js
import React, { useState } from 'react';

const NotificationForm = ({ onSendNotification }) => {
    const [message, setMessage] = useState('');

    const handleSend = () => {
        if (message.trim()) {
            onSendNotification(message);
            setMessage('');
        }
    };

    return (
        <div className="notification-form">
            <input
                type="text"
                placeholder="Type your notification..."
                value={message}
                onChange={(e) => setMessage(e.target.value)}
            />
            <button onClick={handleSend}>Send</button>
        </div>
    );
};

export default NotificationForm;
