import React, { useEffect, useMemo, useState } from "react";
import { io } from "socket.io-client";
import { Button, Container, TextField, Typography, Stack } from "@mui/material";

function App() {
  const socket = useMemo(() => io("http://localhost:3000"), []);

  const [message, SetMessage] = useState("");
  const [room, SetRoom] = useState("");
  const [roomName, SetRoomName] = useState("");
  const [socketID, SetSocketID] = useState("");
  const [messages, SetMessages] = useState([]); // 🔹 Store received messages in an array

  // Function to join a room
  const joinRoom = () => {
    if (room.trim() !== "") {
      socket.emit("join-room", room);
      console.log(`Joined room: ${room}`);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (room.trim() === "") {
      console.log("⚠️ Error: Room ID is required!");
      return;
    }

    socket.emit("message", { message, room }); // ✅ Corrected "Room" -> "room"
    SetMessage("");
  };

  useEffect(() => {
    socket.on("connect", () => {
      SetSocketID(socket.id);
      console.log("Connected", socket.id);
    });

    socket.on("receive-message", (data) => {
      console.log("📩 Received Message:", data);
      SetMessages((prevMessages) => [...prevMessages, data]); // 🔹 Append message to list
    });

    socket.on("welcome", (s) => {
      console.log(s);
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  return (
    <Container maxWidth="sm">
      <Typography variant="h2" component="div" gutterBottom>
        Chat App
      </Typography>

      <Typography variant="h6" component="div" gutterBottom>
        Socket ID: {socketID}
      </Typography>

      {/* Room Input */}
      <TextField
        value={room}
        onChange={(e) => SetRoom(e.target.value)}
        label="Room ID"
        variant="outlined"
      />
      <Button onClick={joinRoom} variant="contained" color="secondary">
        Join Room
      </Button>

      

      {/* Message Form */}
      <form onSubmit={handleSubmit}>
        <TextField
          value={message}
          onChange={(e) => SetMessage(e.target.value)}
          label="Message"
          variant="outlined"
        />
        <Button type="submit" variant="contained" color="primary">
          Send
        </Button>
      </form>

      {/* Messages List */}
      <Stack spacing={2} mt={3}>
        {messages.map((m, i) => (
          <Typography key={i} variant="h6" gutterBottom>
            {m}
          </Typography>
        ))}
      </Stack>
    </Container>
  );
}

export default App;
