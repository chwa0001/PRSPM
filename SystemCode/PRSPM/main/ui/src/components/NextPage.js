import React ,{useState,useEffect} from 'react';
import Avatar from '@material-ui/core/Avatar';
import Button from '@material-ui/core/Button';
import CssBaseline from '@material-ui/core/CssBaseline';
import TextField from '@material-ui/core/TextField';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Checkbox from '@material-ui/core/Checkbox';
import Link from '@material-ui/core/Link';
import Grid from '@material-ui/core/Grid';
import Box from '@material-ui/core/Box';
import LockOutlinedIcon from '@material-ui/icons/LockOutlined';
import Typography from '@material-ui/core/Typography';
import { makeStyles } from '@material-ui/core/styles';
import Container from '@material-ui/core/Container';
import { useHistory } from "react-router-dom";
import FormControl from '@material-ui/core/FormControl';
import Select from '@material-ui/core/Select';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import Cookies from 'js-cookie';
import Fade from '@material-ui/core/Fade';
import CircularProgress from '@material-ui/core/CircularProgress';

const useStyles = makeStyles((theme) => ({
  paper: {
    marginTop: theme.spacing(0),
    display: 'flex',
    flexDirection: 'column',
    alignItems:'center'
  },
  avatar: {
    margin: theme.spacing(1),
    backgroundColor: theme.palette.secondary.main,
  },
  form: {
    width: '100%', // Fix IE 11 issue.
    marginTop: theme.spacing(0),
  },
  submit: {
    backgroundColor:"#0a57f2",
    color: 'white',
    height: 36,
    margin: theme.spacing(3, 0, 2),
  },
  formControl: {
    display: 'flex',
    margin: theme.spacing(1),
    minWidth: 25,
  },
}));

export default function NextPage() {
  const classes = useStyles();
  const [doses,setDoses] = useState(0);
  const [vaxManu,setVaxManu] = useState('');
  const [numdays,setNumdays] = useState(0);
  // const [hosdays,setHosdays] = useState(0);
  const [birthDefect, setBirthDefect] = useState(false);
  const [disability, setDisability] = useState(false);
  const [visit, setVisit] = useState(false);
  const [desease, setDesease] = useState(false);
  const [loading,setLoading] = useState(true);

  let history = useHistory();
  const [userStatus, setUserStatus] = useState(-1);
  const patientID = Cookies.get('id');
  useEffect(() => {
    if(userStatus===0)
    {
      setUserStatus(-1);
      setLoading(false);
      alert('Prediction has been executed!');
      history.push('/final');
    }
    else{
      setLoading(false);
    }
  }, [userStatus])
  function Proceed(patientID,vaxManu,doses,numdays,birthDefect,disability,visit,desease){
  setLoading(true);
  let vaxManuName = ''
  switch(vaxManu) {
    case 'M':
      vaxManuName = 'MODERNA';
      break;
    case 'P':
      vaxManuName = 'PFIZER\\BIONTECH';
      break;
    case 'J':
      vaxManuName = 'JANSSEN';
      break;
    default:
      vaxManuName = 'UNKNOWN MANUFACTURER';
      break;
    }
  const requestOptions = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      patientID:patientID,
      vaccineBrand:vaxManuName,
      numDoses:doses,
      numDays:numdays,
      // numHosDays:hosdays,
      birthDefect:birthDefect,
      disability:disability,
      visit:visit,
      desease:desease,
    }),
  };
  fetch('/SaveVaccination',requestOptions)
  .then(function(response){
    if(!response.ok){
      return response.json().then(text => {throw new Error(text['Bad Request'])})
    }
    else{
      return response.json()
    }
  }).then(
    (data) => {
      setUserStatus(0)
    },(error) => {
      setUserStatus(1)
      alert(error);
    }
  )
}

  const handleDropDownVax = (event) => {
    setVaxManu(event.target.value)
    };
  return (
    <Container component="main" maxWidth="md">
      <CssBaseline />
      <div className={classes.paper}>
        <Avatar className={classes.avatar}>
          <LockOutlinedIcon />
        </Avatar>
        <Typography component="h1" variant="h5">
          Please answer the following question..
        </Typography>
        <form className={classes.form} noValidate>
          <Grid container spacing={2}>
            <Grid item xs={12} container direction="row" spacing={2}>
              <Grid item xs>
              <FormControl 
              fullWidth
              className={classes.formControl}
              >
                <TextField
                  id="doses"
                  label="Number of Doses Taken"
                  type="number"            
                  inputProps={{
                    step: 1,
                    min:1,
                    max:120,
                  }}
                  style={{ paddingTop: '10px' ,paddingRight: '0px'}}
                  value={doses}
                  fullWidth
                  onChange={e => setDoses(e.target.value)}
                />
              </FormControl>
              </Grid>
              <Grid item xs>
              <FormControl 
              fullWidth
              className={classes.formControl}
              >
                <TextField
                  id="numdays"
                  label="Number of Days after Vaccination"
                  type="number"            
                  inputProps={{
                    step: 1,
                    min:1,
                    max:120,
                  }}
                  style={{ paddingTop: '10px' ,paddingRight: '0px'}}
                  value={numdays}
                  fullWidth
                  onChange={e => setNumdays(e.target.value)}
                />
              </FormControl>
              </Grid>
              
            </Grid>
            <Grid item xs={12} container direction="row" spacing={2}>
            <Grid item xs>
              <FormControl 
              fullWidth
              className={classes.formControl}
              >
                <InputLabel id="vaxManu">Manufacturer</InputLabel>
                <Select
                  labelId="vaxManu"
                  id="vaxManu"
                  value={vaxManu}
                  onChange={handleDropDownVax}
                >
                  <MenuItem value="">
                    <em>None</em>
                  </MenuItem>
                  <MenuItem value='M'>MODERNA</MenuItem>
                  <MenuItem value='P'>PFIZER, BIONTECH</MenuItem>
                  <MenuItem value='J'>JANSSEN</MenuItem>
                  <MenuItem value='U'>UNKNOWN MANUFACTURER</MenuItem>
                </Select>
              </FormControl>
              </Grid>
              {/* <Grid item xs>
              <FormControl 
              fullWidth
              className={classes.formControl}
              >
                <TextField
                  id="hosdays"
                  label="Number of Days on hospitalization after Vaccination"
                  type="number"            
                  inputProps={{
                    step: 1,
                    min:1,
                    max:120,
                  }}
                  style={{ paddingTop: '20px' ,paddingRight: '0px'}}
                  value={hosdays}
                  fullWidth
                  onChange={e => setHosdays(e.target.value)}
                />
              </FormControl>
              </Grid> */}
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                variant = 'outlined'
                control={<Checkbox checked={birthDefect} onChange={(e)=>{setBirthDefect(e.target.checked)}} color="primary" />}
                label={`I had ${birthDefect?"":" no "}congenital anomaly or birth defect associated with the
                vaccination.`}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                variant = 'outlined'
                control={<Checkbox checked={disability} onChange={(e)=>{setDisability(e.target.checked)}} color="primary" />}
                label={`I was ${disability?"":" not "}disabled as a result of the vaccination`}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={<Checkbox checked={visit} onChange={(e)=>{setVisit(e.target.checked)}} color="primary" />}
                label={`I had ${visit?"":" not "}visited doctor or other healthcare provider office/clinic.`}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={<Checkbox checked={desease} onChange={(e)=>{setDesease(e.target.checked)}} color="primary" />}
                label={`I had ${desease?" ":" not "} suffered any life-treatening desease before.`}
              />
            </Grid>
          </Grid>
          <Grid item xs={12}>
          <Button
            fullWidth
            variant="contained"
            color="primary"
            className={classes.submit}
            onClick={() => Proceed(patientID,vaxManu,doses,numdays,birthDefect,disability,visit,desease)}
            disabled={loading}
          >
            Get my results!
          </Button>
          {loading && (
          <CircularProgress
            size={24}
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              marginTop: '-12px',
              marginLeft: '-12px',
            }}
          />
          )}
          </Grid>
        </form>
      </div>
    </Container>
  );
}