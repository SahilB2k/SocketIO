import React, { useEffect, useMemo, useState } from 'react'
import {io} from "socket.io-client"
import {Button, Container, TextField, Typography} from "@mui/material"


function App() {
  
  const socket=useMemo(()=>io("http://localhost:3000"), []) 


  const [message,SetMessage]=useState("")
  const [Room,SetRoom]=useState("")
  const [socketID,SetSocketID]=useState("")
  const handleSubmit=(e)=>{
    e.preventDefault();
    socket.emit("message",{message,Room})
    SetMessage("")

  }

  useEffect(()=>{
    socket.on("connect",()=>{
      SetSocketID(socket.id)
     console.log("Connected",socket.id) 
    })

    socket.on("receive-message",(data)=>{
      console.log(data)
    })



    socket.on("welcome",(s)=>{
      console.log(s)
    })

    return ()=>{
      socket.disconnect();

    }
    
  },[]);
  return (
    <Container maxWidth="sm">
      <Typography variant='h2' component="div" gutterBottom>
       
      </Typography>
    
      <Typography variant='h6' component="div" gutterBottom>
        {socketID}
      </Typography>




      <form onSubmit={handleSubmit}>
        <TextField value={message} onChange={e=>SetMessage(e.target.value)} id="outlined-basic" label="Message" variant="outlined">


        </TextField> 
        <TextField value={Room} onChange={e=>SetRoom(e.target.value)} id="outlined-basic" label="Room" variant="outlined">


        </TextField> 
        <Button type='submit' variant='contained' color='primary'>
          Send
        </Button>
      </form>
      
    </Container>
  )
}

export default App
