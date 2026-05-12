const ImageUploader = ({ setImage }) => {

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Validate format
      const validFormats = ['image/jpeg', 'image/png', 'application/dicom'];
      if (!validFormats.includes(file.type) && !file.name.toLowerCase().endsWith('.dcm')) {
        alert('File format error. Only JPG, PNG and DICOM are allowed.');
        return;
      }
      
      // Validate size (10 MB)
      const maxSize = 10 * 1024 * 1024; // 10 MB in bytes
      if (file.size > maxSize) {
        alert('File exceeds the allowed size of 10 MB. Please select a smaller image.');
        return;
      }
      
      // Basic content validation
      if (!file.type.startsWith('image/') && !file.name.toLowerCase().endsWith('.dcm')) {
        alert('Invalid or corrupted image. Please verify the file and try again.');
        return;
      }
      
      setImage(file);
    }
  };

  return (
    <input type="file" accept=".jpg,.jpeg,.png,.dcm,image/*" onChange={handleChange} />
  );
};

export default ImageUploader;