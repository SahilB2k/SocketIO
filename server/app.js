import express from 'express';
import { Server } from 'socket.io';
import { createServer } from 'http';
import cors from 'cors';

const port = 3000;
const app = express();
const server = createServer(app);

// CORS Configuration
const io = new Server(server, {
    cors: {
        origin: "http://localhost:5173",
        methods: ["GET", "POST"],
        credentials: true
    }
});

// Apply CORS to Express
app.use(cors({
    origin: "http://localhost:5173",
    methods: ["GET", "POST"],
    credentials: true
}));

// Basic Route
app.get('/', (req, res) => {
    res.send("Hello world");
});

// Socket.IO Connection
io.on("connection", (socket) => {
    console.log("User Connected:", socket.id);


    socket.on("join-room", (room) => {
        socket.join(room);
        console.log(`User ${socket.id} joined room ${room}`);
    });

    // Listen for messages
    socket.on("message", ({room,message}) => {
        console.log({room,message});
        
        // Emit to all clients
        io.to(room).emit("receive-message", message);
    });

    // Handle Disconnect
    socket.on("disconnect", () => {
        console.log("User Disconnected:", socket.id);
    });
});

// Start the server
server.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
