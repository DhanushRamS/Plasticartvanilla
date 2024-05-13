// src/components/NotificationItem.js
import React from 'react';

const NotificationItem = ({ notification }) => {
    return (
        <div className={`notification-item ${notification.read ? 'read' : 'unread'}`}>
            <span>{notification.message}</span>
            <small>{notification.timestamp}</small>
        </div>
    );
};

export default NotificationItem;
