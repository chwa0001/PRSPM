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
    marginTop: theme.spacing(1),
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
    marginTop: theme.spacing(3),
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
    minWidth: 50,
  },
}));

export default function MainPage() {
  const classes = useStyles();
  const [fullname,setFullname] = useState('');
  const [patientID,setPatientID] = useState(0);
  const [age,setAge] = useState(10);
  const [gender,setGender] = useState('');
  const [symptomText,setSymptomText] = useState('');
  const [checked, setChecked] = useState(true);
  const [loading,setLoading] = useState(true);

  let history = useHistory();
  const [userStatus, setUserStatus] = useState(-1);
  useEffect(() => {
    if(userStatus===0)
    {
      setUserStatus(-1);
      setLoading(false);
      alert('Patient is registered successfully!');
      Cookies.set('id', patientID);
      history.push('/next');
    }
    else{
      setLoading(false);
    }
  }, [userStatus])
  function Proceed(fullname,patientID,age,gender,symptomText){
    if (patientID>0){
      setLoading(true);
      fetch(`/CheckPatientID?id=${patientID}`)
        .then(function(response){
          if(!response.ok){
            return response.json().then(text => {throw new Error(text['Bad Request'])})
          }
          else{
            return response.json()
          }
        }).then(
          (data) => {
            const requestOptions = {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                patientID:patientID,
                name:fullname,
                age:age,
                gender:gender,
                text:symptomText,
              }),
          };
          fetch('/CreateUsers',requestOptions)
          .then(function(response){
            if(!response.ok){
              return response.json().then(text => {throw new Error(text['Bad Request'])})
            }
            else{
              return response.json()
            }
          }).then(
            (data) => {
              console.log(data)
              setUserStatus(0)
            },(error) => {
              setUserStatus(1)
              alert(error);
            }
          )
          },
          (error) => {
            setUserStatus(1)
            alert(error)
          }
        )
    }
    else{
      setUserStatus(1)
      alert('Partient ID is invalid!')
    };

  }

    const handleCheckBoxAgreement = (event) => {
      setChecked(event.target.checked);
    };
    const handleDropDownGender = (event) => {
      switch(event.target.value) {
        case 'M':
          setGender('M')
          break;
        case 'F':
          setGender('F')
          break;
        case 'U':
          setGender('U')
          break;
        default:
          setGender('')
          break;
      }
    };
  return (
    <Container component="main" maxWidth="md">
      <CssBaseline />
      <div className={classes.paper}>
        <Avatar className={classes.avatar}>
          <LockOutlinedIcon />
        </Avatar>
        <Typography component="h1" variant="h5">
          Are you feeling uncomfortable after vaccination?
        </Typography>
        <form className={classes.form} noValidate>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                autoComplete="fname"
                name="fullName"
                variant="outlined"
                required
                fullWidth
                id="fullname"
                label="Full Name"
                autoFocus
                onChange={e => setFullname(e.target.value)}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                variant="outlined"
                required
                fullWidth
                id="patientID"
                label="Patient ID（eg: 1-10000）"
                name="patientID"
                autoComplete="off"
                onChange={e => setPatientID(e.target.value)}
              />
            </Grid>
            <Grid item xs={12} container direction="row" spacing={2}>
              <Grid item xs>
              <FormControl 
              fullWidth
              className={classes.formControl}
              >
                <TextField
                  id="age"
                  label="Age"
                  type="number"            
                  inputProps={{
                    step: 1,
                    min:1,
                    max:120,
                  }}
                  value={age}
                  fullWidth
                  onChange={e => setAge(e.target.value)}
                />
              </FormControl>
              </Grid>
              <Grid item xs>
              <FormControl 
              fullWidth
              className={classes.formControl}
              >
                <InputLabel id="Gender">Gender</InputLabel>
                <Select
                  labelId="Gender"
                  id="Gender"
                  value={gender}
                  onChange={handleDropDownGender}
                >
                  <MenuItem value="">
                    <em>None</em>
                  </MenuItem>
                  <MenuItem value={'M'}>Male</MenuItem>
                  <MenuItem value={'F'}>Female</MenuItem>
                  <MenuItem value={'U'}>Unisex</MenuItem>
                </Select>
              </FormControl>
              </Grid>
            </Grid>
            <Grid item xs={12}>
              <TextField
                variant="outlined"
                required
                fullWidth
                multiline
                minRows={3}
                name="SymptomText"
                label="Describe your symptoms after vaccination."
                type="SymptomText"
                autoComplete="off"
                value={symptomText}
                onChange={e => setSymptomText(e.target.value)}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={<Checkbox checked={checked} onChange={handleCheckBoxAgreement} color="primary" />}
                label="I agree that the information is correct and the data will be used for modeling purpose."
              />
            </Grid>
          </Grid>
          <Grid item xs={12}>
          <Button
            fullWidth
            variant="contained"
            color="primary"
            className={classes.submit}
            onClick={() => Proceed(fullname,patientID,age,gender,symptomText)}
            disabled={loading}
          >
            PROCEED
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