const ImageUploader = ({ setImage }) => {

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
    }
  };

  return (
    <input type="file" accept="image/*" onChange={handleChange} />
  );
};

export default ImageUploader;