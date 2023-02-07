import React ,{useState,useEffect} from 'react';
import Avatar from '@material-ui/core/Avatar';
import CssBaseline from '@material-ui/core/CssBaseline';
import LockOutlinedIcon from '@material-ui/icons/LockOutlined';
import Typography from '@material-ui/core/Typography';
import { makeStyles,styled } from '@material-ui/core/styles';
import Container from '@material-ui/core/Container';
import Grid from '@material-ui/core/Grid';
import { useHistory } from "react-router-dom";
import Slider from '@material-ui/core/Slider';
import FormControl from '@material-ui/core/FormControl';
import Select from '@material-ui/core/Select';
import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import Fade from '@material-ui/core/Fade';
import TextField from '@material-ui/core/TextField';
import Divider from '@material-ui/core/Divider';

const useStyles = makeStyles((theme) => ({
  paper: {
    marginTop: theme.spacing(1),
    display: 'flex',
    flexDirection: 'column',
    alignItems:'center',
  },
  avatar: {
    margin: theme.spacing(1),
    backgroundColor: theme.palette.secondary.main,
  },
  header: {
    marginTop: theme.spacing(0),
    display: 'flex',
    flexDirection: 'column',
    width:'100%',
    alignItems:'center',
    marginBottom: theme.spacing(5),
  },
  form: {
    width: '100%', // Fix IE 11 issue.
    marginTop: theme.spacing(1),
    display: 'flex',
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
    marginBottom:'20px'
  },
}));

const PrettoSlider = styled(Slider)({
  color: '#52af77',
  height: 4,
  flexGrow:1,
  fullWidth:1,
  '& .MuiSlider-track': {
    border: 'none',
    height:5,
  },
  '& .MuiSlider-thumb': {
    height: 15,
    width: 15,
    backgroundColor: '#fff',
    border: '2px solid currentColor',
    '&:focus, &:hover, &.Mui-active, &.Mui-focusVisible': {
      boxShadow: 'inherit',
    },
    '&:before': {
      display: 'none',
    },
  },
  '& .MuiSlider-valueLabel': {
    lineHeight: 0.5,
    fontSize: 10,
    background: 'unset',
    padding: 0,
    width: 0,
    height: 0,
    borderRadius: '50% 50% 50% 50%',
    backgroundColor: '#52af77',
    
  },
  '& .PrivateValueLabel-circle-11': {
    width: '20px',
    height: '20px',
  },
  '& .PrivateValueLabel-offset-10': {
    top:'-15px',
  },
});

export default function PatientDataPage() {
  const classes = useStyles();
  let history = useHistory();
  let jsonPatient = {
        patient: {
            name: '',
            age: '',
            gender: '',
        },
        textSymptoms: {
            text: ''
        },
        vaccinationData: {vaccineBrand: "",numDoses: '',
            numDays: '',
            // numHosDays: '',
            birthDefect: '',
            disability: '',
            visit: '',
            lifeDesease: ''
        },
        symptomsPredicted: {
            symptom: [],
            symptomBert: [],
            symptomTFIDF: [],
            symptomWordVec: []
    }
}

  const [preBert,setPreBertThreshold] = useState(0);
  const [preWordVec,setPreWordVecThreshold] = useState(0);
  const [preTFIDF,setPreTFIDFThreshold] = useState(0);
  const [preHospitalised,setPreHospitalisedThreshold] = useState(0);

  const [Bert,setBertThreshold] = useState(0);
  const [WordVec,setWordVecThreshold] = useState(0);
  const [TFIDF,setTFIDFThreshold] = useState(0);
  const [Hospitalised,setHospitalisedThreshold] = useState(0);
  const [patientIDList,setPatientIDList] = useState([]);
  const [patientID,setPatientID] = useState([]);
  const [patientData,setPatientData] = useState(jsonPatient);

  async function GetThreshold() {
    const response = await fetch(`/CheckModelThreshold`);
    if (!response.ok) {
      const message = `An error has occured: ${response.status}`;
      throw new Error(message);
    }
    else{
      const data = await response.json();
      if(Object.keys(data).length===4){
        setPreBertThreshold(data.Bert*100);
        setPreWordVecThreshold(data.WordVec*100);
        setPreTFIDFThreshold(data.TFIDF*100);
        setPreHospitalisedThreshold(data.Hospitalised*100);

        setBertThreshold(data.Bert*100);
        setWordVecThreshold(data.WordVec*100);
        setTFIDFThreshold(data.TFIDF*100);
        setHospitalisedThreshold(data.Hospitalised*100);
        console.log(data)
      }
      else{
        throw new Error("Invalid threshold data");
      }
    }
  }

  async function GetPatientList() {
    const response = await fetch(`/CheckPatientIDList`);
    if (!response.ok) {
      const message = `An error has occured: ${response.status}`;
      throw new Error(message);
    }
    else{
      const data = await response.json();
      if(data.patientList.length>0){
        setPatientIDList(data.patientList);
      }
      else{
        throw new Error("No patient data found");
      }
    }
  }

  useEffect(() => {
    GetThreshold().catch(console.error);
    GetPatientList().catch(console.error);
  }, []);

  function UpdateThreshold(modelType,newThreshold){
    const requestOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name:modelType,
        thresholdValue:newThreshold/100,
      }),
    };
    fetch('/UpdateThreshold',requestOptions)
    .then(function(response){
      if(!response.ok){
        return response.json().then(text => {throw new Error(text['Bad Request'])})
      }
      else{
        return response.json()
      }
    }).then(
      (data) => {
        switch(modelType) {
          case 'Bert':
            setPreBertThreshold(newThreshold);
            break;
          case 'WordVec':
            setPreWordVecThreshold(newThreshold);
            break;
          case 'TFIDF':
            setPreTFIDFThreshold(newThreshold);
            break;
          case 'Hospitalised':
            setPreHospitalisedThreshold(newThreshold);
            break;
          default:
            console.log('Cannot find correct model type to set the threshold')
            break;
        }
      },(error) => {
        switch(modelType) {
          case 'Bert':
            setBertThreshold(preBert);
            break;
          case 'WordVec':
            setWordVecThreshold(preWordVec);
            break;
          case 'TFIDF':
            setTFIDFThreshold(preTFIDF);
            break;
          case 'Hospitalised':
            setHospitalisedThreshold(preHospitalised);
            break;
          default:
            console.log('Cannot find correct model type to set the threshold')
            break;
        }
        alert(error);
      }
    )
  }

  const handleDropDownPatientID = (event) => {
    setPatientID(event.target.value);
    fetch(`/CheckPatientData?id=${event.target.value}`)
        .then(function(response){
          if(!response.ok){
            return response.json().then(text => {throw new Error(text['Bad Request'])})
          }
          else{
            return response.json()
          }
        }).then(
          (data) => {
            setPatientData(data.patientData)
          }),
          (error) => {
            alert(error)
          }
  };
  
  return (
    <Container component="main" maxWidth="md">
      <CssBaseline />
      <div className={classes.paper}>
        <form className={classes.header} noValidate>
        <Avatar className={classes.avatar}>
          <LockOutlinedIcon />
        </Avatar>
        <Typography component="h1" variant="h5">
          Please review the patient data and threshold for model prediction.
        </Typography>
        </form>
        <form className={classes.form} noValidate>
        <Grid container spacing={2}>
        <Grid item xs={12}>
        <Typography style={{fontSize:'20px',fontWeight: 600}}>
          Custom tune model prediction threshold:
        </Typography>
        </Grid>
        <Grid item xs={12}>
        <Typography style={{fontSize:'15px',fontWeight: 600}}>
          Symptom prediction
        </Typography>
        </Grid>
        <Grid item xs={12} container direction="row" >
        <Grid item xs={3}>
        <Typography style={{fontSize:'15px',fontWeight: 400}}>
          Threshold for Bert Model:
        </Typography>
        </Grid>
        <Grid item xs={9}>
        <PrettoSlider
        valueLabelDisplay="auto"
        aria-label="pretto slider"
        defaultValue={Bert}
        max={100}
        min={0}
        value={Bert}
        onChange={(event,value)=>{setBertThreshold(value)}}
        onChangeCommitted={()=>{UpdateThreshold('Bert',Bert)}}
        />
        </Grid>
        </Grid>
        <Grid item xs={12} container direction="row" >
        <Grid item xs={3}>
        <Typography style={{fontSize:'15px',fontWeight: 400}}>
          Threshold for WordVec Model:
        </Typography>
        </Grid>
        <Grid item xs={9}>
        <PrettoSlider
        valueLabelDisplay="auto"
        aria-label="pretto slider"
        defaultValue={WordVec}
        max={100}
        min={0}
        value={WordVec}
        onChange={(event,value)=>{setWordVecThreshold(value)}}
        onChangeCommitted={()=>{UpdateThreshold('WordVec',WordVec)}}
        />
        </Grid>
        </Grid>
        <Grid item xs={12} container direction="row" >
        <Grid item xs={3}>
        <Typography style={{fontSize:'15px',fontWeight: 400}}>
          Threshold for TFIDF Model:
        </Typography>
        </Grid>
        <Grid item xs={9}>
        <PrettoSlider
        valueLabelDisplay="auto"
        aria-label="pretto slider"
        defaultValue={TFIDF}
        max={100}
        min={0}
        value={TFIDF}
        onChange={(event,value)=>{setTFIDFThreshold(value)}}
        onChangeCommitted={()=>{UpdateThreshold('TFIDF',TFIDF)}}
        />
        </Grid>
        </Grid>
        <Grid item xs>
        <Typography style={{fontSize:'15px',fontWeight: 600}}>
          Hospitalisation prediction
        </Typography>
        </Grid>
        <Grid item xs={12} container direction="row" >
        <Grid item xs={3}>
        <Typography style={{fontSize:'15px',fontWeight: 400}}>
          Threshold for Hospitalised Model:
        </Typography>
        </Grid>
        <Grid item xs={9}>
        <PrettoSlider
        valueLabelDisplay="auto"
        aria-label="pretto slider"
        defaultValue={Hospitalised}
        max={100}
        min={0}
        value={Hospitalised}
        onChange={(event,value)=>{setHospitalisedThreshold(value)}}
        onChangeCommitted={()=>{UpdateThreshold('Hospitalised',Hospitalised)}}
        />
        </Grid>
        </Grid>
        <Grid item xs>
        <Divider/>
        </Grid>
        </Grid>
        </form>
        <form className={classes.form} noValidate>
        <Grid item xs={12} container direction="row" spacing={2}>
        <Grid item xs>
        <Typography style={{fontSize:'20px',fontWeight: 600}}>
          Patient Database
        </Typography>
        </Grid>
        <Grid item xs={12} >
          <FormControl 
            fullWidth
            className={classes.formControl}
            >
              <InputLabel id="patientID">Patient ID</InputLabel>
              <Select
                id="patientID"
                value={patientID}
                onChange={handleDropDownPatientID}
              >
                <MenuItem value="">
                  <em>Select any patient ID</em>
                </MenuItem>
                {patientIDList.map((patientID) => (<MenuItem key={patientID.toString()} value={patientID.toString()}>{patientID}</MenuItem>))
                }
              </Select>
          </FormControl>
        </Grid>
        <Fade in={patientID!=0} unmountOnExit>        
        <Grid item xs={12} container direction="row" spacing={2} alignItems='flex-start'>
        <Grid item xs={4}>
        <TextField
                name="fullName"
                variant="outlined"
                fullWidth
                id="fullname"
                label="Full Name"
                value={patientData.patient.name}
                InputProps={{
                  readOnly: true,
                }}
              />
        </Grid>
        <Grid item xs={2}>
        <TextField
                name="age"
                variant="outlined"
                fullWidth
                id="age"
                label="Age"
                value={patientData.patient.age}
                InputProps={{
                  readOnly: true,
                }}
              />
        </Grid>
        <Grid item xs={2}>
        <TextField
                name="gender"
                variant="outlined"
                fullWidth
                id="gender"
                label="Gender"
                value={patientData.patient.gender==='M'?patientData.patient.gender==='F'?'Female':'Male':'Unknown'}
                InputProps={{
                  readOnly: true,
                }}
              />
        </Grid>
        <Grid item xs={12} container direction="row" spacing={2} alignItems='flex-start'>
        <Grid item xs={4}>
        <TextField
                name="vaccineBrand"
                variant="outlined"
                fullWidth
                id="vaccineBrand"
                label="Vaccine Manufacturer"
                value={patientData.vaccinationData.vaccineBrand}
                InputProps={{
                  readOnly: true,
                }}
              />
        </Grid>
        <Grid item xs={3}>
        <TextField
                name="numDoses"
                variant="outlined"
                fullWidth
                id="numDoses"
                label="Number of doses taken"
                value={patientData.vaccinationData.numDoses}
                InputProps={{
                  readOnly: true,
                }}
              />
        </Grid>
        <Grid item xs={3}>
        <TextField
                name="numDays"
                variant="outlined"
                fullWidth
                id="numDays"
                label="Number of days after vaccination"
                value={patientData.vaccinationData.numDays}
                InputProps={{
                  readOnly: true,
                }}
              />
        </Grid>
        <Grid item xs={12} container direction="row" spacing={2} alignItems='flex-start'>        
        <Grid item xs>
        <TextField
                name="text"
                variant="outlined"
                fullWidth
                multiline
                minRows={3}
                id="text"
                label="Symptoms described by patient"
                value= {patientData.textSymptoms.text}
                InputProps={{
                  readOnly: true,
                }}
              />
        </Grid>
        {/* <Grid item xs={6}>
        <TextField
                name="numHosDays"
                variant="outlined"
                fullWidth
                id="numHosDays"
                label="Number of days on hospitalization after vaccination"
                value={patientData.vaccinationData.numHosDays}
                InputProps={{
                  readOnly: true,
                }}
              />
        </Grid> */}
        </Grid>
        </Grid>
        <Grid item xs={12} container direction="column" spacing={1} alignItems='flex-start'>
        <Grid item xs>
        <Typography style={{fontSize:'15px',fontWeight: 200, margin:10}}>
          1. The patient had{patientData.vaccinationData.birthDefect?" ":" no "}congenital anomaly or birth defect associated with the vaccination.
        </Typography>
        </Grid>
        <Grid item xs>
        <Typography style={{fontSize:'15px',fontWeight: 200, margin:10}}>
          2. The patient was{patientData.vaccinationData.disability?" ":" not "}disabled as a result of the vaccination
        </Typography>
        </Grid>
        <Grid item xs>
        <Typography style={{fontSize:'15px',fontWeight: 200, margin:10}}>
          3. The patient had{patientData.vaccinationData.visit?" ":" not "}visited doctor or other healthcare provider office/clinic.
        </Typography>
        </Grid>
        <Grid item xs>
        <Typography style={{fontSize:'15px',fontWeight: 200, margin:10}}>
          4. The patient had{patientData.vaccinationData.desease?" ":" not "}suffered any life-treatening desease before.
        </Typography>
        </Grid>
        </Grid>
        <Grid item xs={12} container direction="row" spacing={1} alignItems='flex-start'>        
        <Grid item xs={6} >
          <FormControl 
            fullWidth
            className={classes.formControl}
            >
              <InputLabel id="hybrid" style={{marginBottom:2}}>Symptoms finalised from hybrid method</InputLabel>
              <Select
                id="hybrid"
              >
                {patientData.symptomsPredicted.symptom.map((symptom) => (<MenuItem key={symptom.toString()} value={symptom.toString()}>{symptom}</MenuItem>))
                }
              </Select>
          </FormControl>
        </Grid>
        <Grid item xs={6} >
          <FormControl 
            fullWidth
            className={classes.formControl}
            >
              <InputLabel id="TFIDF" style={{marginBottom:2}}>Symptoms predicted from TFIDF Model</InputLabel>
              <Select
                id="TFIDF"
              >
                {patientData.symptomsPredicted.symptomTFIDF.map((symptom) => (<MenuItem key={symptom.toString()} value={symptom.toString()}>{symptom}</MenuItem>))
                }
              </Select>
          </FormControl>
        </Grid>
        </Grid>
        <Grid item xs={12} container direction="row" spacing={1} alignItems='flex-start'>        
        <Grid item xs={6} >
          <FormControl 
            fullWidth
            className={classes.formControl}
            >
              <InputLabel id="Bert" style={{marginBottom:2}}>Symptoms predicted from Bert Model</InputLabel>
              <Select
                id="Bert"
              >
                {patientData.symptomsPredicted.symptomBert.map((symptom) => (<MenuItem key={symptom.toString()} value={symptom.toString()}>{symptom}</MenuItem>))
                }
              </Select>
          </FormControl>
        </Grid>
        <Grid item xs={6} >
          <FormControl 
            fullWidth
            className={classes.formControl}
            >
              <InputLabel id="WordVec" style={{marginBottom:2}}>Symptoms predicted from WordVec Model</InputLabel>
              <Select
                id="WordVec"
              >
                {patientData.symptomsPredicted.symptomWordVec.map((symptom) => (<MenuItem key={symptom.toString()} value={symptom.toString()}>{symptom}</MenuItem>))
                }
              </Select>
          </FormControl>
        </Grid>
        </Grid>

        </Grid>
        </Fade>
      </Grid>
      </form>
      </div>
    </Container>
  );
}