// src/components/NotificationList.js
import React from 'react';
import NotificationItem from './NotificationItem';

const NotificationList = ({ notifications }) => {
    return (
        <div className="notification-list">
            {notifications.map((notification) => (
                <NotificationItem key={notification.id} notification={notification} />
            ))}
        </div>
    );
};

export default NotificationList;
